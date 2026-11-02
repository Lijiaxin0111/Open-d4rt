"""A3 detection rows and A4 pose baselines on synthetic tables (l3d.check_rows)."""

from __future__ import annotations

import numpy as np
import pytest

from l3d import check_rows as cr
from l3d import detection as det
from l3d import geometry as geo
from l3d import sensitivity as sens
from l3d.decompose import table_scale

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)
MAX_TRACKS = 32
TOL = 1e-4  # p_xyz is float32


@pytest.fixture(scope="module")
def scene() -> sens.Scene:
    return sens.make_scene(0, **SMALL)


def _rows(scene: sens.Scene, inj: sens.Injection | None = None, model_scale: float = 0.5) -> tuple[dict, dict]:
    dec = sens.VirtualDecoder(scene) if inj is None else sens.VirtualDecoder(scene, inj, seed=7)
    d = sens.synthetic_table(scene, dec, max_tracks=MAX_TRACKS, model_scale=model_scale)
    return d, cr.row_errors(d, table_scale(d))


def test_clean_decoder_gives_zero_residuals(scene: sens.Scene) -> None:
    d, err = _rows(scene)
    assert np.nanmax(err["rel"]) < TOL  # the median scale undoes model_scale exactly
    K = cr.fit_K(d, err)
    ss = cr.source_swap_detection(d, err)
    cs = cr.camera_swap_detection(d, err)
    rc = cr.ray_center_detection(d, err, K)
    sr = cr.static_rigidity_detection(d, err)
    for name, res in (("ss model", ss["model_u2"]["residual"]), ("ss gt", ss["gt_u2"]["residual"]),
                      ("cs se3", cs["residual_se3"]), ("cs sim3", cs["residual_sim3"]), ("cs target", cs["target"]),
                      ("rc gt", rc["residual_gt"]), ("sr", sr["residual"])):
        assert res.size > 0 and np.nanmax(res) < TOL, name
    for out in (cs, rc, sr, ss["model_u2"]):
        n = out["residual" if "residual" in out else "residual_se3"].size
        assert all(np.asarray(out[k]).size == n for k in ("error", "target", "conf"))


def test_clean_decoder_recovers_poses(scene: sens.Scene) -> None:
    d, err = _rows(scene)
    pb = cr.pose_baselines(d, err)
    assert pb["pairs"]["t"].size > 0 and np.all(pb["pairs"]["t"] != pb["pairs"]["c"])
    assert np.max(pb["pairs"]["gt_rot_deg"]) > 1.0  # the scene camera rotates
    for m in ("umeyama", "umeyama_conf", "ray_center_gtK"):
        assert np.max(pb["methods"][m]["rot_deg"]) < 1e-2, m
        assert np.max(pb["methods"][m]["trans_rel"]) < 1e-3, m
    assert pb["trajectory"]["umeyama"]["ate_rel"] < 1e-3
    assert np.max(pb["pairs"]["cs_residual"]) < TOL


def test_checks_see_their_injected_component(scene: sens.Scene) -> None:
    _, clean = _rows(scene)
    d, err = _rows(scene, sens.Injection("nonrigid", "C", 0.05))
    cs = cr.camera_swap_detection(d, err)
    assert np.mean(cs["residual_sim3"]) > 1e-3
    assert det.spearman(cs["residual_sim3"], cs["target"]) > 0.9  # Sim(3) camera swap = non-rigid error
    d, err = _rows(scene, sens.Injection("depth_jitter", "S", 0.05))
    ss = cr.source_swap_detection(d, err)["gt_u2"]
    assert np.mean(ss["residual"]) > 1e-3 and det.spearman(ss["residual"], ss["target"]) > 0.5


def test_nearest_rotation_and_rigid_relative_pose() -> None:
    rng = np.random.default_rng(0)
    R = geo.rotvec_to_matrix(rng.normal(size=(5, 3)) * 0.3)
    noisy = R + 1e-3 * rng.normal(size=R.shape)
    Rn = geo.nearest_rotation(noisy)
    np.testing.assert_allclose(Rn @ np.swapaxes(Rn, -1, -2), np.broadcast_to(np.eye(3), Rn.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(Rn), 1.0, atol=1e-12)
    assert np.max(geo.rotation_angle_deg(np.swapaxes(Rn, -1, -2) @ R)) < 0.2
    T = np.tile(np.eye(4), (5, 1, 1))
    T[:, :3, :3] = noisy
    rel = geo.relative_pose(T, np.arange(5), np.zeros(5, dtype=int))[:, :3, :3]
    np.testing.assert_allclose(rel @ np.swapaxes(rel, -1, -2), np.broadcast_to(np.eye(3), rel.shape), atol=1e-12)
