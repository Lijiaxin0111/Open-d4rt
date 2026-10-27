from __future__ import annotations

import numpy as np
import pytest

from l3d import geometry as geo


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    return geo.rotvec_to_matrix(rng.normal(size=3))


@pytest.mark.parametrize("with_scale", [True, False])
def test_umeyama_recovers_sim3_and_se3(with_scale: bool) -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3)) * [2.0, 1.0, 3.0] + [0.0, 0.0, 5.0]
    R = _rand_rot(rng)
    s = 1.7 if with_scale else 1.0
    t = rng.normal(size=3)
    dst = geo.apply_sim3(src, s, R, t)
    s_hat, R_hat, t_hat = geo.umeyama(src, dst, with_scale=with_scale)
    assert s_hat == pytest.approx(s, rel=1e-10)
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-10)


def test_umeyama_planar_points_need_reflection_fix() -> None:
    # Coplanar points: the SVD can return a reflection; the fit must still be the rotation.
    rng = np.random.default_rng(1)
    src = np.c_[rng.normal(size=(40, 2)), np.zeros(40)]
    for _ in range(20):
        R = _rand_rot(rng)
        dst = src @ R.T + [1.0, 2.0, 3.0]
        s, R_hat, t = geo.umeyama(src, dst, with_scale=True)
        assert np.linalg.det(R_hat) == pytest.approx(1.0)
        np.testing.assert_allclose(R_hat, R, atol=1e-9)
        assert s == pytest.approx(1.0)


def test_umeyama_mirrored_data_returns_proper_rotation() -> None:
    rng = np.random.default_rng(2)
    src = rng.normal(size=(30, 3))
    dst = src * [-1.0, 1.0, 1.0]  # a pure reflection: no rotation fits exactly
    _, R_hat, _ = geo.umeyama(src, dst, with_scale=False)
    assert np.linalg.det(R_hat) == pytest.approx(1.0)
    np.testing.assert_allclose(R_hat @ R_hat.T, np.eye(3), atol=1e-12)


def test_umeyama_weights_and_nonfinite_rows() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(60, 3))
    R = _rand_rot(rng)
    dst = src @ R.T + 0.5
    dst[:10] += rng.normal(size=(10, 3)) * 5.0  # outliers
    w = np.ones(60)
    w[:10] = 0.0
    dst[10] = np.nan
    _, R_hat, t_hat = geo.umeyama(src, dst, weights=w, with_scale=False)
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, 0.5, atol=1e-10)
    s, _, _ = geo.umeyama(src[:2], dst[:2])
    assert np.isnan(s)


def test_relative_pose_and_inverse() -> None:
    rng = np.random.default_rng(4)
    T_w2c = np.tile(np.eye(4), (5, 1, 1))
    for i in range(5):
        T_w2c[i, :3, :3] = _rand_rot(rng)
        T_w2c[i, :3, 3] = rng.normal(size=3)
    np.testing.assert_allclose(geo.invert_se3(T_w2c) @ T_w2c, np.tile(np.eye(4), (5, 1, 1)), atol=1e-12)
    Xw = rng.normal(size=3)
    X2 = geo.apply_transform(T_w2c[2], Xw)
    X4 = geo.apply_transform(geo.relative_pose(T_w2c, 2, 4), X2)
    np.testing.assert_allclose(X4, geo.apply_transform(T_w2c[4], Xw), atol=1e-12)
    rel = geo.relative_pose(T_w2c, np.array([0, 1]), np.array([3, 3]))
    assert rel.shape == (2, 4, 4)


def test_project_unproject_and_pixel_normalization() -> None:
    K = np.array([[220.0, 0, 127.5], [0, 210.0, 127.5], [0, 0, 1]])
    rng = np.random.default_rng(5)
    uv = rng.uniform(0, 255, size=(20, 2))
    depth = rng.uniform(2, 8, size=20)
    X = geo.unproject(K, uv, depth)
    np.testing.assert_allclose(X[:, 2], depth)
    np.testing.assert_allclose(geo.project(K, X), uv, atol=1e-10)
    assert np.isnan(geo.project(K, np.array([0.0, 0.0, -1.0]))).all()
    hw = (240, 320)
    uvn = geo.uv_px_to_norm(uv, hw)
    np.testing.assert_allclose(geo.uv_norm_to_px(uvn, hw), uv)
    np.testing.assert_allclose(geo.uv_norm_to_px(np.array([1.0, 1.0]), hw), [319.0, 239.0])
    d = geo.ray_dirs(K, uv)
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0)
    np.testing.assert_allclose(np.cross(d, X), 0.0, atol=1e-10)


def test_ls_ray_center_recovers_camera_center() -> None:
    rng = np.random.default_rng(6)
    o = np.array([0.3, -0.2, 1.5])
    dirs = rng.normal(size=(100, 3)) + [0, 0, 3]
    lam = rng.uniform(2, 8, size=100)
    pts = o + lam[:, None] * dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    center, dist = geo.ls_ray_center(pts, dirs)
    np.testing.assert_allclose(center, o, atol=1e-9)
    assert dist.max() < 1e-9
    # Batched, with a broken line ignored through NaN.
    pts_b = np.stack([pts, pts + 1.0])
    pts_b[1, 0] = np.nan
    centers, _ = geo.ls_ray_center(pts_b, np.stack([dirs, dirs]))
    np.testing.assert_allclose(centers, [o, o + 1.0], atol=1e-9)
    # Perturbed points: positive spread.
    _, dist_n = geo.ls_ray_center(pts + rng.normal(size=pts.shape) * 0.05, dirs)
    assert dist_n.mean() > 0.01


def test_fit_intrinsics_from_points_exact() -> None:
    K = np.array([[220.0, 0, 127.5], [0, 205.0, 95.5], [0, 0, 1]])
    rng = np.random.default_rng(7)
    X = np.c_[rng.uniform(-2, 2, 200), rng.uniform(-1.5, 1.5, 200), rng.uniform(2, 8, 200)]
    K_hat = geo.fit_intrinsics_from_points(X, geo.project(K, X), (192, 256))
    np.testing.assert_allclose(K_hat, K, atol=1e-9)


def test_median_scale_and_rotation_angle() -> None:
    rng = np.random.default_rng(8)
    gt = rng.normal(size=(100, 3)) + [0, 0, 5]
    pred = gt / 3.0
    pred[0] = np.nan
    assert geo.median_scale(gt, pred) == pytest.approx(3.0, rel=1e-3)
    R = geo.rotvec_to_matrix(np.array([0.0, np.radians(30.0), 0.0]))
    assert float(geo.rotation_angle_deg(R)) == pytest.approx(30.0)
