from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from l3d import decompose as dec
from l3d import geometry as geo
from l3d import sensitivity as sens
from l3d import table


def _small_scene(seed: int = 0) -> sens.Scene:
    return sens.make_scene(seed, T=12, n_static=120, n_objects=2, n_per_object=40)


def _planted(seed: int, weights: tuple[float, float, float], n_groups: int = 6, T: int = 40, per_group: int = 300):
    """e_d = offset_g + jitter_gt + noise, with jitter centered over t and noise over each cell."""
    rng = np.random.default_rng(seed)
    so, sj, sn = weights
    labels = np.repeat(np.arange(n_groups), per_group)
    off = rng.normal(size=n_groups) * so
    jit = rng.normal(size=(T, n_groups)) * sj
    jit -= jit.mean(0)
    noise = rng.normal(size=(T, labels.size)) * sn
    for g in range(n_groups):
        cols = labels == g
        noise[:, cols] -= noise[:, cols].mean(1, keepdims=True)
    e = off[labels][None] + jit[:, labels] + noise
    n_tot = float(e.size)
    planted = {
        "offset": float((off[labels] ** 2).sum() * T / (e**2).sum()),
        "jitter": float((jit[:, labels] ** 2).sum() / (e**2).sum()),
        "residual": float((noise**2).sum() / (e**2).sum()),
    }
    return e, labels, planted, n_tot


@pytest.mark.parametrize("weights", [(1.0, 0.5, 0.3), (0.2, 1.0, 0.5), (0.3, 0.3, 1.0)])
def test_offset_jitter_split_recovers_planted_shares(weights: tuple[float, float, float]) -> None:
    e, labels, planted, _ = _planted(0, weights)
    out = dec.offset_jitter_split(e, labels)
    for k in planted:
        assert out["shares"][k] == pytest.approx(planted[k], abs=0.02)
    assert sum(out["shares"].values()) == pytest.approx(1.0, abs=1e-12)


def test_offset_jitter_split_noisy_draws_within_2pct() -> None:
    # Uncentered draws: leakage of noise into group means is ~1/per_group, well under 2%.
    rng = np.random.default_rng(1)
    n_groups, T, per_group = 8, 40, 400
    labels = np.repeat(np.arange(n_groups), per_group)
    off = rng.normal(size=n_groups) * 0.8
    jit = rng.normal(size=(T, n_groups)) * 0.6
    noise = rng.normal(size=(T, labels.size)) * 0.7
    e = off[labels][None] + jit[:, labels] + noise
    out = dec.offset_jitter_split(e, labels)
    eff_off = off + jit.mean(0)  # what the estimator can see as offset
    total = (e**2).sum()
    assert out["shares"]["offset"] == pytest.approx((eff_off[labels] ** 2).sum() * T / total, abs=0.02)
    assert out["shares"]["jitter"] == pytest.approx(((jit - jit.mean(0)) ** 2).sum() * per_group / total, abs=0.02)
    assert out["shares"]["residual"] == pytest.approx((noise**2).sum() / total, abs=0.02)


@pytest.mark.parametrize("weighting", ["points", "cells"])
def test_offset_jitter_split_sums_to_one_with_missing(weighting: str) -> None:
    rng = np.random.default_rng(2)
    e = rng.normal(size=(20, 90)) + 0.3
    e[rng.random(e.shape) < 0.3] = np.nan
    labels = rng.integers(-1, 5, size=90)
    out = dec.offset_jitter_split(e, labels, weighting=weighting)
    assert sum(out["shares"].values()) == pytest.approx(1.0, abs=1e-12)
    assert out["residual"].shape == e.shape


def test_self_frame_errors_depth_and_image_plane() -> None:
    rng = np.random.default_rng(3)
    X = np.c_[rng.uniform(-1, 1, 30), rng.uniform(-1, 1, 30), rng.uniform(2, 6, 30)]
    out = dec.self_frame_errors(X * np.exp(0.1), X, K=np.diag([200.0, 200.0, 1.0]))
    np.testing.assert_allclose(out["e_d"], 0.1, atol=1e-12)
    np.testing.assert_allclose(out["angle"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["px"], 0.0, atol=1e-9)
    r = X / np.linalg.norm(X, axis=-1, keepdims=True)
    perp = np.cross(r, [0.0, 1.0, 0.0])
    perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
    out = dec.self_frame_errors(X + 0.05 * np.linalg.norm(X, axis=-1, keepdims=True) * perp, X)
    np.testing.assert_allclose(out["e_d"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["angle"], 0.05, atol=1e-12)
    np.testing.assert_allclose(out["sq_err"], out["radial"] ** 2 + (out["perp"] ** 2).sum(-1), atol=1e-12)


def test_group_tracks_rigid_recovers_objects_and_knn_patches() -> None:
    scene = _small_scene(0)
    labels = dec.group_tracks(scene.X_world, scene.is_static, mode="rigid")
    assert (labels[scene.is_static] == 0).all()
    for g in range(1, int(scene.group.max()) + 1):
        lab = np.unique(labels[scene.group == g])
        assert lab.size == 1 and lab[0] > 0
    assert np.unique(labels[~scene.is_static]).size == int(scene.group.max())
    knn = dec.group_tracks(scene.X_world, scene.is_static, mode="knn", k=16)
    sizes = np.bincount(knn[~scene.is_static])[1:]
    assert sizes.size == round((~scene.is_static).sum() / 16)
    assert sizes.min() >= 1


def test_camera_frame_error_gauge_vs_nonrigid() -> None:
    rng = np.random.default_rng(4)
    Q = np.c_[rng.uniform(-2, 2, 200), rng.uniform(-1, 1, 200), rng.uniform(2, 8, 200)]
    R = geo.rotvec_to_matrix(np.array([0.0, 0.05, 0.0]))
    P = geo.apply_sim3(Q, 1.03, R, np.array([0.1, 0.0, 0.0]))
    pair = np.repeat([0, 1], 100)
    out = dec.camera_frame_error(P, Q, pair, pair + 5)
    assert out["nonrigid_share"] < 1e-12
    np.testing.assert_allclose(out["gauge_rot_deg"], np.degrees(0.05), atol=1e-8)
    np.testing.assert_allclose(out["gauge_log_scale"], np.log(1.03), atol=1e-10)
    out = dec.camera_frame_error(P + rng.normal(size=P.shape) * 0.2, Q, pair, pair + 5)
    assert 0.0 < out["nonrigid_share"] < 1.0


def test_table_roundtrip_validate_and_from_table(tmp_path: Path) -> None:
    scene = _small_scene(1)
    decoder = sens.VirtualDecoder(scene, sens.Injection("depth_offset", "T", 0.05), seed=3)
    d = sens.synthetic_table(scene, decoder, model_scale=0.37)
    path = tmp_path / "clip.npz"
    np.savez(path, **d)
    loaded = table.load_table(path)
    assert table.validate_table(loaded) == []
    assert isinstance(loaded["sequence"], str) and isinstance(loaded["T"], int)
    bad = dict(loaded)
    bad["p_xyz"] = bad["p_xyz"][:, :2]
    del bad["g_vis"]
    with pytest.raises(ValueError, match="p_xyz"):
        table.validate_table(bad)
    assert len(table.validate_table(bad, strict=False)) == 2
    sel = table.select(loaded, design="cross", self_frame=False)
    assert (sel["design"] == 1).all() and (sel["q_tc"] != sel["q_tt"]).all()

    a1 = dec.from_table(loaded)
    assert a1["scale"] == pytest.approx(1.0 / 0.37, rel=0.06)
    # A per-object depth offset is pure offset (the global scale only shifts it).
    assert a1["depth_split"]["shares"]["offset"] == pytest.approx(1.0, abs=1e-4)
    # It propagates rigidly into cross-frame outputs: no non-rigid camera-frame error. The
    # global scale does not undo the model scale exactly (s * m != 1), which leaves a
    # per-pair translation (s m - 1) t_rel: pure gauge.
    cam = a1["camera_frame"]
    assert np.max(cam["nonrigid_rms_rel"]) < 1e-5
    assert np.max(cam["total_rms_rel"]) > 1e-3 and cam["gauge_share"] > 0.999

    gauge = sens.synthetic_table(scene, sens.VirtualDecoder(scene, sens.Injection("gauge", "C"), seed=4))
    cam = dec.from_table(gauge)["camera_frame"]
    assert cam["gauge_share"] > 0.999
    assert np.median(cam["gauge_rot_deg"]) > 0.5
