"""Visibility / motion / parallax splits (l3d.splits) on exact geometry and synthetic tables."""

from __future__ import annotations

import numpy as np

from l3d import sensitivity as sens
from l3d import splits
from l3d.table import select

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)


def test_parallax_is_the_baseline_angle_at_the_point() -> None:
    T_w2c = np.tile(np.eye(4), (2, 1, 1))
    T_w2c[1, 0, 3] = -1.0  # camera 1 center at x = +1 (world = camera 0)
    X = np.array([0.0, 0.0, 1.0])
    gt = np.stack([X, X - np.array([1.0, 0.0, 0.0])])[:, None]  # [T=2, N=1, 3], own-camera coordinates
    d = {"T_w2c_gt": T_w2c, "gt_xyz_cam": gt}
    one = np.array([0])
    np.testing.assert_allclose(splits.parallax_deg(d, one, one, np.array([1])), [45.0], atol=1e-9)
    np.testing.assert_allclose(splits.parallax_deg(d, one, np.array([1]), np.array([1])), [0.0], atol=1e-9)
    np.testing.assert_array_equal(splits.parallax_bin(np.array([0.0, 0.99, 1.0, 4.9, 5.0, 80.0, np.nan])),
                                  [0, 0, 1, 1, 2, 2, -1])


def test_clean_decoder_is_perfect_in_every_bin_and_bins_cover_all_rows() -> None:
    scene = sens.make_scene(0, **SMALL)
    d = sens.synthetic_table(scene, sens.VirtualDecoder(scene), max_tracks=32, model_scale=0.5)
    pooled = splits.pool([splits.clip_sums(d)])
    summary = splits.summarize(pooled)
    r = select(d, design="track", has_gt=True)
    n_cam0 = int((r["q_tc"] == 0).sum())
    for split in ("visibility", "motion"):
        assert sum(b["track_rows"] for b in summary[split]) == n_cam0
    assert sum(b["track_rows"] for b in summary["parallax"]) <= n_cam0
    used = [b for s in summary.values() for b in s if b["track_rows"] > 0]
    assert used and all(b["apd"] == 1.0 and b["epe"] < 1e-4 for b in used)
    assert sum(b["track_rows"] for b in summary["parallax"][1:]) > 0  # the synthetic camera translates
    cells = [b for s in summary.values() for b in s if b["tsc_cells"] > 0]
    assert cells and all(b["tsc_mean_total"] < 1e-8 for b in cells)
