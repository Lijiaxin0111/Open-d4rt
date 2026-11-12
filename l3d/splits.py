"""Part A numbers split by GT visibility, motion and parallax (one decoded table at a time).

The proposal evaluates "per subset and split by visibility, motion and parallax". Labels
come from ground truth only:

- **visibility**: GT visibility of the point at the target frame ``t_t`` (``g_vis``);
- **motion**: the dump's ``is_static`` (GT world displacement over the clip below 1% of the
  median depth) -> static or dynamic;
- **parallax**: the angle at the GT point ``X(t_t)`` between the rays from the camera centers
  of ``t_s`` and ``t_t``, binned by :data:`PARALLAX_EDGES` (degrees). It is 0 for ``t_s = t_t``
  and for a fixed camera (pstudio). A design-1 cell (one point and target frame, several
  source frames) takes the median over its source frames.

:func:`clip_sums` returns additive sums per split and bin, so clips and subsets pool by
addition (as in A1 and A3): tracking APD/EPE on design-0 rows decoded in camera 0 (the
upstream protocol inside the clip), the depth share of the self-frame error (design 0,
``t_c = t_t``), and the T/S/C parts of the 3D error over complete design-1 cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import geometry as geo
from . import tsc
from .decompose import self_frame_errors, table_scale
from .gauge import THRESHOLDS
from .table import select

PARALLAX_EDGES = (0.0, 1.0, 5.0, np.inf)
SPLITS = {
    "visibility": ("visible", "occluded"),
    "motion": ("static", "dynamic"),
    "parallax": tuple(f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}"
                      for lo, hi in zip(PARALLAX_EDGES[:-1], PARALLAX_EDGES[1:])),
}
TSC_PARTS = ("T", "S", "C", "SC", "total")


def parallax_deg(d: dict[str, Any], track: np.ndarray, ts: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Angle (deg) at GT point ``track`` at time ``tt`` between the rays from cameras ``ts`` and ``tt``."""
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    c2w = geo.invert_se3(geo.rigidify(T_w2c))
    C = c2w[:, :3, 3]
    X_cam = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[tt, track]
    Xw = np.einsum("mij,mj->mi", c2w[tt, :3, :3], X_cam) + c2w[tt, :3, 3]
    a, b = Xw - C[ts], Xw - C[tt]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))  # stable near 0


def parallax_bin(angle: np.ndarray) -> np.ndarray:
    """Bin index of each angle in :data:`PARALLAX_EDGES` (``-1`` for non-finite)."""
    idx = np.digitize(angle, PARALLAX_EDGES[1:-1])
    return np.where(np.isfinite(angle), idx, -1)


def _labels(d: dict[str, Any], track: np.ndarray, ts_or_par: np.ndarray, tt: np.ndarray,
            vis: np.ndarray, parallax: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Bin index per split (``-1`` = unlabeled) for rows or cells."""
    static = np.asarray(d["is_static"], dtype=bool)[track]
    par = parallax_deg(d, track, ts_or_par, tt) if parallax is None else parallax
    return {"visibility": np.where(vis, 0, 1), "motion": np.where(static, 0, 1), "parallax": parallax_bin(par)}


def _add(acc: dict[str, list[dict[str, float]]], labels: dict[str, np.ndarray], values: dict[str, np.ndarray]) -> None:
    for split, lab in labels.items():
        for b in range(len(SPLITS[split])):
            m = lab == b
            for k, v in values.items():
                acc[split][b][k] = acc[split][b].get(k, 0.0) + float(np.nansum(v[m]))


def empty_sums() -> dict[str, list[dict[str, float]]]:
    return {split: [{} for _ in bins] for split, bins in SPLITS.items()}


def clip_sums(d: dict[str, Any], scale: float | None = None) -> dict[str, list[dict[str, float]]]:
    """Additive sums per split and bin for one decoded table (see the module docstring)."""
    if scale is None:
        scale = table_scale(d)
    acc = empty_sums()

    # Tracking: design 0 decoded in camera 0 (world = clip frame 0), the upstream protocol.
    r = select(d, design="track", has_gt=True)
    cam0 = r["q_tc"] == 0
    tr, ts, tt = (r[k][cam0].astype(np.int64) for k in ("q_track", "q_ts", "q_tt"))
    err = np.linalg.norm(r["p_xyz"][cam0].astype(np.float64) * scale - r["g_xyz"][cam0].astype(np.float64), axis=-1)
    ok = np.isfinite(err)
    apd = np.mean([err <= th for th in THRESHOLDS], axis=0)
    labels = _labels(d, tr, ts, tt, r["g_vis"][cam0].astype(bool))
    _add(acc, labels, {"track_n": ok.astype(float), "track_apd": np.where(ok, apd, 0.0), "track_epe": np.where(ok, err, 0.0)})

    # Self-frame error (t_c = t_t): depth (radial) share of the squared error, relative error.
    own = r["q_tc"] == r["q_tt"]
    sf = self_frame_errors(r["p_xyz"][own].astype(np.float64) * scale, r["g_xyz"][own].astype(np.float64))
    tr2, ts2, tt2 = (r[k][own].astype(np.int64) for k in ("q_track", "q_ts", "q_tt"))
    ok2 = np.isfinite(sf["sq_err"]) & np.isfinite(sf["radial"])
    dist = np.linalg.norm(r["g_xyz"][own].astype(np.float64), axis=-1)
    labels = _labels(d, tr2, ts2, tt2, r["g_vis"][own].astype(bool))
    _add(acc, labels, {"self_n": ok2.astype(float), "self_sq": np.where(ok2, sf["sq_err"], 0.0),
                       "self_radial_sq": np.where(ok2, sf["radial"] ** 2, 0.0),
                       "self_rel": np.where(ok2, np.sqrt(sf["sq_err"]) / np.maximum(dist, 1e-12), 0.0)})

    # T/S/C of the 3D error over complete design-1 cells; a cell's parallax is its sources' median.
    tf = tsc.tensor_from_table(d, component="xyz", scale=scale)
    sp = tsc.two_way_split(tf["E"], tf["mask"])
    n_i, t_i = np.nonzero(sp["complete"])
    if n_i.size:
        track, t = tf["tracks"][n_i], tf["t_frames"][t_i]
        src = tf["s_frames"][n_i, t_i]  # [cells, S], -1 on unused slots
        active = sp["row_active"][n_i, t_i] & (src >= 0)
        rep = np.repeat(np.arange(n_i.size), src.shape[1])
        ang = parallax_deg(d, track[rep], np.maximum(src.reshape(-1), 0), t[rep]).reshape(src.shape)
        cell_par = np.nanmedian(np.where(active, ang, np.nan), axis=1)
        vis = np.asarray(d["gt_vis"], dtype=bool)[t, track]
        labels = _labels(d, track, src[:, 0], t, vis, parallax=cell_par)
        vals = {f"tsc_{k}": sp[k][n_i, t_i] for k in TSC_PARTS}
        vals["tsc_cells"] = np.ones(n_i.size)
        _add(acc, labels, vals)
    return acc


def pool(sums: list[dict[str, list[dict[str, float]]]]) -> dict[str, list[dict[str, float]]]:
    """Add per-clip sums."""
    out = empty_sums()
    for s in sums:
        for split, bins in s.items():
            for b, vals in enumerate(bins):
                for k, v in vals.items():
                    out[split][b][k] = out[split][b].get(k, 0.0) + v
    return out


def summarize(pooled: dict[str, list[dict[str, float]]]) -> dict[str, list[dict[str, Any]]]:
    """Metrics per split and bin from pooled sums (NaN where a bin is empty)."""

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    out: dict[str, list[dict[str, Any]]] = {}
    for split, bins in pooled.items():
        n_track = sum(v.get("track_n", 0.0) for v in bins)
        n_cells = sum(v.get("tsc_cells", 0.0) for v in bins)
        rows = []
        for name, v in zip(SPLITS[split], bins):
            tot = v.get("tsc_total", 0.0)
            rows.append({
                "bin": name,
                "track_rows": int(v.get("track_n", 0)), "track_frac": ratio(v.get("track_n", 0.0), n_track),
                "apd": ratio(v.get("track_apd", 0.0), v.get("track_n", 0.0)),
                "epe": ratio(v.get("track_epe", 0.0), v.get("track_n", 0.0)),
                "self_rows": int(v.get("self_n", 0)),
                "self_rel_err": ratio(v.get("self_rel", 0.0), v.get("self_n", 0.0)),
                "depth_share": ratio(v.get("self_radial_sq", 0.0), v.get("self_sq", 0.0)),
                "tsc_cells": int(v.get("tsc_cells", 0)), "cell_frac": ratio(v.get("tsc_cells", 0.0), n_cells),
                **{f"{k}_share": ratio(v.get(f"tsc_{k}", 0.0), tot) for k in ("T", "S", "C", "SC")},
                "tsc_mean_total": ratio(tot, v.get("tsc_cells", 0.0)),
            })
        out[split] = rows
    return out
