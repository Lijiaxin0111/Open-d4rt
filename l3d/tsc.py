"""A3: T/S/C split of the error over source (S) and camera (C) frames.

For each point ``n`` and target frame ``t`` decoded on a crossed grid of source slots
``s`` and camera slots ``c``, the balanced two-way decomposition of ``E[n, t, s, c, :]``

    E = mu + a_s + b_c + i_sc,   mean_sc |E|^2 = |mu|^2 + mean_s |a_s|^2 + mean_c |b_c|^2 + mean_sc |i_sc|^2

gives the shared part T = ``|mu|^2``, the S and C main effects and their interaction SC.
The identity is exact for complete grids, so cells with a missing entry are dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from l3d import geometry as geo
from l3d.decompose import table_scale
from l3d.table import select

_EPS = 1e-12


def _as5(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 4:
        E = E[..., None]
    if E.ndim != 5:
        raise ValueError(f"E must be [n, t, s, c] or [n, t, s, c, d], got {E.shape}")
    return E


def _shares(parts: dict[str, np.ndarray], total: np.ndarray) -> dict[str, float]:
    tot = float(np.nansum(total))
    return {k: (float(np.nansum(v)) / tot if tot > 0 else float("nan")) for k, v in parts.items()}


def two_way_split(E: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Balanced two-way (S x C) split of ``mean |E|^2`` per ``(n, t)`` cell.

    ``E`` is ``[n, t, S, C, d]`` (or ``[n, t, S, C]`` for scalars); ``mask`` ``[n, t, S, C]``
    marks valid entries (default: finite). A cell uses its active source slots (rows with
    any valid entry) and camera slots; it is complete when every active ``(s, c)`` is
    valid, so a track seen in fewer source frames still counts (its own grid is
    balanced). Returns per-cell ``T, S, C, SC, total`` (``[n, t]``, NaN for incomplete
    cells), ``shares`` = sum of part / sum of totals over complete cells, ``n_cells`` and
    the effects ``mu [n,t,d]``, ``a [n,t,S,d]``, ``b [n,t,C,d]`` (0 on inactive slots).
    """
    E = _as5(E)
    fin = np.isfinite(E).all(-1)
    mask = fin if mask is None else (np.asarray(mask, dtype=bool) & fin)
    row_act = mask.any(axis=3)
    col_act = mask.any(axis=2)
    ns = row_act.sum(-1).astype(np.float64)
    nc = col_act.sum(-1).astype(np.float64)
    complete = (mask.sum(axis=(2, 3)) == ns * nc) & (ns > 0) & (nc > 0)
    nsc = np.maximum(ns * nc, 1.0)
    E0 = np.where(mask[..., None], E, 0.0)
    mu = E0.sum(axis=(2, 3)) / nsc[..., None]
    a = np.where(row_act[..., None], E0.sum(axis=3) / np.maximum(nc, 1.0)[..., None, None] - mu[:, :, None], 0.0)
    b = np.where(col_act[..., None], E0.sum(axis=2) / np.maximum(ns, 1.0)[..., None, None] - mu[:, :, None], 0.0)
    inter = np.where(mask[..., None], E0 - mu[:, :, None, None] - a[:, :, :, None] - b[:, :, None, :], 0.0)

    nan = np.where(complete, 1.0, np.nan)
    out = {
        "T": (mu**2).sum(-1) * nan,
        "S": (a**2).sum(axis=(-1, -2)) / np.maximum(ns, 1.0) * nan,
        "C": (b**2).sum(axis=(-1, -2)) / np.maximum(nc, 1.0) * nan,
        "SC": (inter**2).sum(axis=(-1, -2, -3)) / nsc * nan,
        "total": (E0**2).sum(axis=(-1, -2, -3)) / nsc * nan,
    }
    out["shares"] = _shares({k: out[k] for k in ("T", "S", "C", "SC")}, out["total"])
    out["n_cells"] = int(complete.sum())
    out["complete"] = complete
    out["row_active"] = row_act
    out["mu"], out["a"], out["b"] = mu, a, b
    return out


def one_way_split(E: np.ndarray, mask: np.ndarray | None = None, min_sources: int = 2) -> dict[str, Any]:
    """S-only split for self-frame components (``c = t`` fixed).

    ``E`` is ``[n, t, S, d]`` (or ``[n, t, S]``). Per cell with at least ``min_sources``
    valid sources: ``T = |mean_s E|^2``, ``S = mean_s |E - mean|^2`` and
    ``mean_s |E|^2 = T + S`` exactly (unequal source counts are fine here).
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 3:
        E = E[..., None]
    fin = np.isfinite(E).all(-1)
    mask = fin if mask is None else (np.asarray(mask, dtype=bool) & fin)
    cnt = mask.sum(-1).astype(np.float64)
    ok = cnt >= min_sources
    E0 = np.where(mask[..., None], E, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = E0.sum(2) / cnt[..., None]
        dev = np.where(mask[..., None], E - mu[:, :, None], 0.0)
        S = (dev**2).sum(-1).sum(-1) / cnt
        total = (E0**2).sum(-1).sum(-1) / cnt
    nan = np.where(ok, 1.0, np.nan)
    out = {"T": (mu**2).sum(-1) * nan, "S": S * nan, "total": total * nan}
    out["shares"] = _shares({"T": out["T"], "S": out["S"]}, out["total"])
    out["n_cells"] = int(ok.sum())
    out["mu"] = mu
    return out


def s_share_vs_gap(
    E: np.ndarray,
    mask: np.ndarray | None,
    s_frames: np.ndarray,
    t_frames: np.ndarray,
    bins: np.ndarray | None = None,
) -> dict[str, Any]:
    """S share binned by ``|t_s - t_t|``.

    ``E`` is ``[n, t, S, C, d]`` (two-way, complete cells only) or ``[n, t, S, d]``
    (one-way). Each ``(n, t, s)`` row contributes its S effect ``|a_s|^2`` and its row
    total ``mean_c |E_sc|^2``; a bin's share is the ratio of their sums, so the bins
    aggregate to the overall S share. ``s_frames`` is ``[n, t, S]``, ``t_frames`` ``[n, t]``
    or ``[t]``. Default bins: ``[0, 1, 2, 4, 8, 16, 32, 64)``.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 4:  # one-way [n, t, S, d] -> two-way with C = 1
        fin = np.isfinite(E).all(-1)
        m = fin if mask is None else (np.asarray(mask, dtype=bool) & fin)
        cnt = m.sum(-1)
        E0 = np.where(m[..., None], E, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = E0.sum(2) / cnt[..., None]
        a2 = np.where(m, ((E - mu[:, :, None]) ** 2).sum(-1), np.nan)
        row_tot = np.where(m, (E**2).sum(-1), np.nan)
        keep = m & (cnt >= 2)[..., None]
    else:
        res = two_way_split(E, mask)
        a2 = (res["a"] ** 2).sum(-1)
        E5 = _as5(E)
        valid = np.isfinite(E5).all(-1) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(E5).all(-1))
        nc = np.maximum(valid.any(axis=2).sum(-1), 1)[..., None]
        row_tot = (np.where(valid[..., None], E5, 0.0) ** 2).sum(axis=(-1, -2)) / nc
        keep = res["complete"][..., None] & res["row_active"]
    s_frames = np.asarray(s_frames)
    t_frames = np.asarray(t_frames)
    if t_frames.ndim == 1:
        t_frames = np.broadcast_to(t_frames[None], s_frames.shape[:2])
    gap = np.abs(s_frames.astype(np.int64) - t_frames[..., None].astype(np.int64))
    bins = np.array([0, 1, 2, 4, 8, 16, 32, 64]) if bins is None else np.asarray(bins)
    which = np.digitize(gap, bins) - 1
    n_bins = bins.size - 1
    share = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    s_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = keep & (which == i)
        count[i] = int(sel.sum())
        if count[i] == 0:
            continue
        tot = float(np.nansum(row_tot[sel]))
        share[i] = float(np.nansum(a2[sel]) / tot) if tot > 0 else np.nan
        s_mean[i] = float(np.nanmean(a2[sel]))
    return {"bin_edges": bins, "share": share, "count": count, "s_effect_mean": s_mean}


def tensor_from_table(
    d: dict[str, Any],
    component: str = "xyz",
    design: int | str = "cross",
    scale: float | None = None,
) -> dict[str, Any]:
    """Arrange design-1 rows into ``E[n, t, s, c, d]`` slots for :func:`two_way_split`.

    Slots: tracks and target frames of the design; per ``(n, t)`` cell, sources and
    cameras sorted ascending. ``component``:
    - ``"xyz"``: 3D error rotated into camera ``t_t`` (GT rotation) / GT depth ``z_t``;
    - ``"log_depth"``: ``log(P . r / |X|)`` along camera ``t_c``'s ray (scalar);
    - ``"image_plane"``: component perpendicular to that ray, rotated into camera ``t_t``, / ``|X|``.
    Outputs are scaled by the clip's global median scale first (computed if ``None``).
    """
    if scale is None:
        scale = table_scale(d)
    r = select(d, design=design, has_gt=True)
    P = r["p_xyz"].astype(np.float64) * scale
    X = r["g_xyz"].astype(np.float64)
    tr, tt, ts, tc = (r[k].astype(np.int64) for k in ("q_track", "q_tt", "q_ts", "q_tc"))
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    R_ct = geo.relative_pose(T_w2c, tc, tt)[:, :3, :3]  # camera t_c -> camera t_t rotation
    z_t = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[tt, tr, 2]

    if component == "xyz":
        val = np.einsum("mij,mj->mi", R_ct, P - X) / z_t[:, None]
    elif component in ("log_depth", "image_plane"):
        dist = np.linalg.norm(X, axis=-1)
        ray = X / np.maximum(dist, _EPS)[:, None]
        along = (P * ray).sum(-1)
        if component == "log_depth":
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(along > 0, np.log(np.maximum(along, _EPS) / dist), np.nan)[:, None]
        else:
            perp = P - along[:, None] * ray
            val = np.einsum("mij,mj->mi", R_ct, perp) / dist[:, None]
    else:
        raise ValueError(f"unknown component '{component}'")

    tracks, tr_i = np.unique(tr, return_inverse=True)
    targets, tt_i = np.unique(tt, return_inverse=True)
    cell = tr_i * targets.size + tt_i

    def _slots(frames: np.ndarray) -> tuple[np.ndarray, int]:
        pair = np.stack([cell, frames], -1)
        uniq, inv = np.unique(pair, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        first = np.r_[0, np.flatnonzero(np.diff(uniq[:, 0])) + 1]
        rank = np.arange(uniq.shape[0]) - np.repeat(first, np.diff(np.r_[first, uniq.shape[0]]))
        return rank[inv], int(rank.max()) + 1 if rank.size else 0

    s_slot, n_s = _slots(ts)
    c_slot, n_c = _slots(tc)
    shape = (tracks.size, targets.size, n_s, n_c)
    E = np.full(shape + (val.shape[1],), np.nan)
    mask = np.zeros(shape, dtype=bool)
    s_frames = np.full(shape[:3], -1, dtype=np.int64)
    c_frames = np.full((shape[0], shape[1], n_c), -1, dtype=np.int64)
    E[tr_i, tt_i, s_slot, c_slot] = val
    mask[tr_i, tt_i, s_slot, c_slot] = np.isfinite(val).all(-1)
    s_frames[tr_i, tt_i, s_slot] = ts
    c_frames[tr_i, tt_i, c_slot] = tc
    return {
        "E": E,
        "mask": mask,
        "tracks": tracks,
        "t_frames": targets,
        "s_frames": s_frames,
        "c_frames": c_frames,
        "scale": scale,
        "component": component,
    }
