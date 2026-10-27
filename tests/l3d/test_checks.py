from __future__ import annotations

import numpy as np
import pytest

from l3d import checks
from l3d import geometry as geo

K = np.array([[220.0, 0, 127.5], [0, 220.0, 127.5], [0, 0, 1]])


def _frame(seed: int = 0, n: int = 300):
    """Frame-t pixels with depths, and a GT pose t -> c."""
    rng = np.random.default_rng(seed)
    uv = rng.uniform(5, 250, size=(n, 2))
    X_t = geo.unproject(K, uv, rng.uniform(2, 8, size=n))
    T = geo.sim3_matrix(1.0, geo.rotvec_to_matrix(np.array([0.02, -0.15, 0.01])), np.array([0.6, 0.05, -0.2]))
    return uv, X_t, T


def test_source_swap_and_psi() -> None:
    rng = np.random.default_rng(1)
    P = rng.normal(size=(50, 3)) + [0, 0, 5]
    assert np.abs(checks.source_swap(P, P)).max() == 0.0
    r = checks.source_swap(P, P + [0.0, 0.5, 0.0])
    np.testing.assert_allclose(r, 0.5 / P[:, 2])
    # Depth-normalized: shrinking the scene leaves it unchanged.
    np.testing.assert_allclose(checks.source_swap(0.1 * P, 0.1 * (P + [0.0, 0.5, 0.0])), r)
    x = np.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(checks.robust_psi(x, "huber", 1.0), [0.0, 0.125, 1.5])
    assert checks.robust_psi(x, "cauchy", 1.0)[2] == pytest.approx(0.5 * np.log(5.0))


@pytest.mark.parametrize("with_scale", [False, True])
def test_camera_swap_recovers_pose(with_scale: bool) -> None:
    _, X_t, T = _frame()
    s = 1.1 if with_scale else 1.0
    P_c = s * geo.apply_transform(T, X_t)
    out = checks.camera_swap(X_t, P_c, with_scale=with_scale)
    assert out["residual"].max() < 1e-10
    np.testing.assert_allclose(out["R"], T[:3, :3], atol=1e-10)
    # SE(3) cannot absorb a scale: residual > 0.
    if with_scale:
        assert checks.camera_swap(X_t, P_c, with_scale=False)["residual"].mean() > 1e-3


def test_ray_center_recovers_center_and_sees_image_plane_error() -> None:
    uv, X_t, T = _frame(2)
    P_c = geo.apply_transform(T, X_t)
    out = checks.ray_center(uv, P_c, K, T[:3, :3], T[:3, 3])
    np.testing.assert_allclose(out["center"], T[:3, 3], atol=1e-9)
    assert out["spread"].max() < 1e-10 and out["center_dist"] < 1e-10
    # Moving points along their rays changes nothing.
    P_far = geo.apply_transform(T, X_t * 1.3)
    assert checks.ray_center(uv, P_far, K, T[:3, :3], T[:3, 3])["spread"].max() < 1e-10
    # Moving them across the rays does.
    rng = np.random.default_rng(3)
    P_off = geo.apply_transform(T, X_t + rng.normal(size=X_t.shape) * 0.1)
    assert checks.ray_center(uv, P_off, K, T[:3, :3], T[:3, 3])["spread"].mean() > 0.005


def test_reprojection_and_static_rigidity() -> None:
    uv, X_t, _ = _frame(4)
    assert checks.reprojection(X_t, K, uv).max() < 1e-9
    P = np.repeat(X_t[:20, None], 6, axis=1)
    out = checks.static_rigidity(P)
    assert np.abs(out["residual"]).max() < 1e-12
    P2 = P.copy()
    P2[:, 0, 0] += 0.6  # one of 6 frames moves by 0.6 m
    res = checks.static_rigidity(P2)["residual"]
    np.testing.assert_allclose(res[:, 0], 0.5 / P.mean(1)[:, 2], rtol=1e-6)
    mask = np.ones(P2.shape[:2], dtype=bool)
    mask[:, 0] = False
    assert np.nanmax(checks.static_rigidity(P2, mask)["residual"]) < 1e-12


def test_pose_from_grid_methods_and_errors() -> None:
    uv, X_t, T = _frame(5)
    P_c = geo.apply_transform(T, X_t)
    rng = np.random.default_rng(6)
    bad = rng.random(X_t.shape[0]) < 0.2
    P_c_noisy = P_c.copy()
    P_c_noisy[bad] += rng.normal(size=(bad.sum(), 3)) * 1.0
    conf = np.where(bad, -8.0, 8.0)
    poses = checks.pose_from_grid(X_t, P_c_noisy, conf=conf, uv_px=uv, K=K)
    err = {k: checks.pose_errors(v["T"], T) for k, v in poses.items()}
    assert err["umeyama_conf"]["rot_deg"] < 0.05 and err["umeyama_conf"]["trans"] < 0.01
    assert err["umeyama"]["trans"] > err["umeyama_conf"]["trans"]
    clean = checks.pose_from_grid(X_t, P_c, uv_px=uv, K=K)
    for v in clean.values():
        e = checks.pose_errors(v["T"], T)
        assert e["rot_deg"] < 1e-6 and e["trans"] < 1e-8
    R10 = geo.sim3_matrix(1.0, geo.rotvec_to_matrix(np.array([0, 0, np.radians(10.0)])), np.zeros(3))
    assert float(checks.pose_errors(R10, np.eye(4))["rot_deg"]) == pytest.approx(10.0)


def test_ate_rpe() -> None:
    rng = np.random.default_rng(7)
    c2w = np.tile(np.eye(4), (10, 1, 1))
    c2w[:, :3, 3] = np.cumsum(rng.normal(size=(10, 3)) * 0.2, axis=0)
    c2w[:, :3, :3] = geo.rotvec_to_matrix(np.linspace(0, 0.3, 10)[:, None] * [0, 1, 0])
    # A Sim(3)-transformed copy (scale 0.5) has zero ATE/RPE after Sim(3) alignment.
    G = geo.sim3_matrix(1.0, geo.rotvec_to_matrix(np.array([0.1, 0.2, 0.3])), np.array([1.0, 0, 0]))
    pred = G @ c2w
    pred[:, :3, 3] *= 0.5
    out = checks.ate_rpe(pred, c2w, align="sim3")
    assert out["ate"] < 1e-9 and out["rpe_trans"] < 1e-9 and out["rpe_rot_deg"] < 1e-6
    assert out["align_scale"] == pytest.approx(2.0)
    assert checks.ate_rpe(pred, c2w, align="se3")["ate"] > 0.01
