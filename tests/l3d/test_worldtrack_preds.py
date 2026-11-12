"""In-image query rule of the WorldTrack scorer and the prediction-saving hooks of the Gate 0 wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "l3d"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_in_image_half_pixel_borders() -> None:
    score = _load("score_worldtrack_preds")
    uv = np.array([[-0.5, -0.5], [639.5, 479.5], [-0.51, 10], [10, 479.51], [1e6, 3], [320, 240]])
    np.testing.assert_array_equal(score.in_image(uv, (480, 640)), [True, True, False, False, False, True])


def test_hooks_save_the_scored_predictions(tmp_path: Path) -> None:
    wrap = _load("run_opend4rt_worldtrack_preds")
    up = wrap.upstream
    T, N = 4, 5
    rng = np.random.default_rng(0)
    xyz = rng.uniform([-1, -1, 1], [1, 1, 3], size=(T, N, 3))
    xyz[0, 3, 2] = np.nan  # dropped by the finite-depth rule
    vis = np.ones((T, N), bool)
    vis[0, 1] = False  # not visible at frame 0
    uv = rng.uniform(0, 100, size=(T, N, 2))
    sample = {"visibility": vis, "tracks_uv": uv, "tracks_xyz_cam": xyz, "video_rgb": np.zeros((T, 8, 8, 3), np.uint8)}
    seen = {}
    orig_load, orig_metrics = up.load_worldtrack_sequence, up._metrics_for_sequence
    up.load_worldtrack_sequence = lambda npz_path, num_frames: sample
    up._metrics_for_sequence = lambda **kw: seen.setdefault("kw", kw) and {"ok": 1}
    try:
        wrap.install_hooks(tmp_path)
        up.load_worldtrack_sequence(tmp_path / "ds_mini" / "seqA.npz", num_frames=T)
        pred = rng.normal(size=(T, 3, 3))
        assert up._metrics_for_sequence(gt_tracks_world=pred, pred_tracks_ref0=pred, compute_dyn=True) == {"ok": 1}
    finally:
        up.load_worldtrack_sequence, up._metrics_for_sequence = orig_load, orig_metrics
    with np.load(tmp_path / "ds_mini" / "seqA.npz") as z:
        np.testing.assert_allclose(z["pred_tracks_ref0"], pred.astype(np.float32))
        np.testing.assert_array_equal(z["query_uv"], uv[0, [0, 2, 4]])
    assert seen["kw"]["compute_dyn"] is True
