"""A1: ground-truth error decomposition (proposal, "Error decomposition").

All functions take model outputs already multiplied by the clip's global median scale
(see :func:`global_scale`); :func:`from_table` does that and reports the scale.

Components:
- self-frame outputs (``t_c == t_t``): depth error ``e_d = log((P . r) / |X|)`` along the
  GT ray ``r = X / |X|`` and image-plane error (component of ``P`` perpendicular to ``r``);
- depth error split per motion group into offset (time mean), jitter (per-frame group
  mean minus offset) and per-point residual;
- camera-frame error of cross-frame outputs vs. their self-frame output mapped by the GT
  pose, split into a best-fit Sim(3) gauge and a non-rigid remainder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from l3d import geometry as geo
from l3d.table import DESIGNS, select

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Self-frame errors
# --------------------------------------------------------------------------------------


def global_scale(p_xyz: np.ndarray, g_xyz: np.ndarray, q_tt: np.ndarray, q_tc: np.ndarray) -> float:
    """Clip scale ``median |X_gt| / median |P|`` over finite self-frame outputs."""
    is_self = np.asarray(q_tt) == np.asarray(q_tc)
    return geo.median_scale(np.asarray(g_xyz)[is_self], np.asarray(p_xyz)[is_self])


def table_scale(d: dict[str, Any]) -> float:
    """:func:`global_scale` of a table: rows with GT and a GT source pixel (design 5 excluded)."""
    r = select(d, has_gt=True, mask=d["design"] != DESIGNS["swap"])
    return global_scale(r["p_xyz"], r["g_xyz"], r["q_tt"], r["q_tc"])


def self_frame_errors(p_xyz: np.ndarray, g_xyz: np.ndarray, K: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Depth and image-plane error of (scaled) self-frame outputs.

    Returns per-row ``e_d`` (log depth ratio along the GT ray; NaN if ``P . r <= 0``),
    ``radial`` (``P . r - |X|``), ``perp`` (``[M, 3]`` component of ``P`` perpendicular to
    ``r``), ``angle`` (``|perp| / |X|``), ``sq_err`` (``|P - X|^2 = radial^2 + |perp|^2``)
    and, with ``K``, ``px`` (pixel distance between the projections of ``P`` and ``X``).
    """
    P = np.asarray(p_xyz, dtype=np.float64)
    X = np.asarray(g_xyz, dtype=np.float64)
    dist = np.linalg.norm(X, axis=-1)
    r = X / np.maximum(dist, _EPS)[..., None]
    along = (P * r).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_d = np.where(along > _EPS, np.log(np.maximum(along, _EPS) / np.maximum(dist, _EPS)), np.nan)
    perp = P - along[..., None] * r
    out = {
        "e_d": e_d,
        "radial": along - dist,
        "perp": perp,
        "angle": np.linalg.norm(perp, axis=-1) / np.maximum(dist, _EPS),
        "sq_err": ((P - X) ** 2).sum(-1),
    }
    if K is not None:
        out["px"] = np.linalg.norm(geo.project(K, P) - geo.project(K, X), axis=-1)
    return out


# --------------------------------------------------------------------------------------
# Motion groups
# --------------------------------------------------------------------------------------


def world_trajectories(gt_xyz_cam: np.ndarray, T_w2c: np.ndarray) -> np.ndarray:
    """GT tracks ``[T, N, 3]`` from own-camera to world (= clip frame 0) coordinates."""
    T_c2w = geo.invert_se3(geo.rigidify(T_w2c))
    return geo.apply_transform(T_c2w[:, None], np.asarray(gt_xyz_cam, dtype=np.float64))


def _pairwise_dist_std(traj: np.ndarray) -> np.ndarray:
    """Std over time of pairwise distances ``[N, N]`` (inf where < 2 common frames)."""
    n = traj.shape[1]
    s1 = np.zeros((n, n))
    s2 = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for x in traj:
        ok = np.isfinite(x).all(-1)
        xz = np.where(ok[:, None], x, 0.0)
        sq = (xz**2).sum(-1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * xz @ xz.T, 0.0))
        both = (ok[:, None] & ok[None, :]).astype(np.float64)
        s1 += d * both
        s2 += d * d * both
        cnt += both
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean**2, 0.0)
    std = np.sqrt(var)
    std[cnt < 2] = np.inf
    np.fill_diagonal(std, 0.0)
    return std


def _first_finite(traj: np.ndarray) -> np.ndarray:
    ok = np.isfinite(traj).all(-1)  # [T, N]
    first = np.argmax(ok, axis=0)
    return traj[first, np.arange(traj.shape[1])]


def group_tracks(
    gt_world_traj: np.ndarray,
    is_static: np.ndarray,
    mode: str = "rigid",
    rel_thresh: float = 0.02,
    k: int = 16,
    ref_depth: float | None = None,
    linkage_method: str = "average",
) -> np.ndarray:
    """Motion-group label per track (``[N]`` int; ``0`` = static, ``-1`` = no finite GT).

    ``"rigid"``: dynamic tracks clustered agglomeratively (``linkage_method``) on the std
    over time of their pairwise 3D distances, cut at ``rel_thresh * ref_depth``.
    ``"knn"``: dynamic tracks split into local patches of about ``k`` points (seeds by
    farthest-point sampling at each track's first finite position, nearest-seed
    assignment), for non-rigid bodies such as people. ``ref_depth`` defaults to the median
    finite ``z`` of the trajectories (world = camera of clip frame 0).
    """
    traj = np.asarray(gt_world_traj, dtype=np.float64)
    is_static = np.asarray(is_static, dtype=bool)
    n = traj.shape[1]
    labels = np.full(n, -1, dtype=np.int64)
    has_gt = np.isfinite(traj).all(-1).any(0)
    labels[is_static & has_gt] = 0
    dyn = np.flatnonzero(~is_static & has_gt)
    if dyn.size == 0:
        return labels
    if dyn.size == 1:
        labels[dyn] = 1
        return labels
    if ref_depth is None:
        z = traj[..., 2]
        ref_depth = float(np.median(np.abs(z[np.isfinite(z)])))

    if mode == "rigid":
        std = _pairwise_dist_std(traj[:, dyn])
        thresh = rel_thresh * ref_depth
        std = np.where(np.isfinite(std), std, 1e6 * max(thresh, _EPS))
        iu = np.triu_indices(dyn.size, k=1)
        Z = linkage(std[iu], method=linkage_method)
        labels[dyn] = fcluster(Z, t=thresh, criterion="distance").astype(np.int64)
    elif mode == "knn":
        pos = _first_finite(traj[:, dyn])
        n_groups = max(1, int(round(dyn.size / float(k))))
        seeds = [int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=-1)))]
        d_min = np.linalg.norm(pos - pos[seeds[0]], axis=-1)
        for _ in range(n_groups - 1):
            nxt = int(np.argmax(d_min))
            seeds.append(nxt)
            d_min = np.minimum(d_min, np.linalg.norm(pos - pos[nxt], axis=-1))
        d_seed = np.linalg.norm(pos[:, None] - pos[seeds][None], axis=-1)
        labels[dyn] = 1 + np.argmin(d_seed, axis=1)
    else:
        raise ValueError(f"unknown grouping mode '{mode}'")
    # Re-label to 0 (static), 1..G (dynamic) without gaps.
    uniq = np.unique(labels[labels > 0])
    remap = {int(u): i + 1 for i, u in enumerate(uniq)}
    labels[labels > 0] = [remap[int(v)] for v in labels[labels > 0]]
    return labels


# --------------------------------------------------------------------------------------
# Offset / jitter / per-point split
# --------------------------------------------------------------------------------------


def offset_jitter_split(e_d: np.ndarray, labels: np.ndarray, weighting: str = "points") -> dict[str, Any]:
    """Split ``mean e_d^2`` into group offset, group jitter and per-point residual.

    ``e_d`` is ``[T, N]`` (NaN = no output), ``labels`` ``[N]`` (negative = ignored).
    ``m[g, t]`` is the group mean of ``e_d`` at frame ``t``; ``offset_g`` its time mean;
    ``jitter[g, t] = m[g, t] - offset_g``; residual ``= e_d - m[g, t]``.

    Weighting (both make the three shares sum to 1 exactly):
    - ``"points"``: every valid ``(n, t)`` counts once; ``offset_g`` is the point-count
      weighted time mean of ``m`` (the plain ``mean_t m[g, t]`` when the group's
      visible count is constant over time).
    - ``"cells"``: every valid ``(g, t)`` cell counts once (points within it equally);
      ``offset_g`` is the plain mean over the group's valid frames.
    """
    e = np.asarray(e_d, dtype=np.float64)
    labels = np.asarray(labels)
    use = labels >= 0
    e = e[:, use]
    lab = labels[use].astype(np.int64)
    n_groups = int(lab.max()) + 1 if lab.size else 0
    valid = np.isfinite(e)
    onehot = np.zeros((lab.size, n_groups))
    onehot[np.arange(lab.size), lab] = 1.0
    counts = valid.astype(np.float64) @ onehot  # [T, G]
    sums = np.where(valid, e, 0.0) @ onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sums / counts
    has = counts > 0
    resid = np.where(valid, e - np.where(has, m, 0.0)[:, lab], np.nan)
    r2 = np.where(valid, resid**2, 0.0)
    e2 = np.where(valid, e**2, 0.0)

    if weighting == "points":
        w = counts
        point_w = np.ones_like(e)
    elif weighting == "cells":
        w = has.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            point_w = np.where(valid, 1.0 / counts[:, lab], 0.0)
    else:
        raise ValueError(f"unknown weighting '{weighting}'")
    m0 = np.where(has, m, 0.0)
    wsum_g = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = (w * m0).sum(0) / wsum_g
    jitter = np.where(has, m0 - offset[None], np.nan)

    total = float((point_w * e2).sum())
    parts = {
        "offset": float((w * np.where(has, offset[None] ** 2, 0.0)).sum()),
        "jitter": float((w * np.where(has, jitter, 0.0) ** 2).sum()),
        "residual": float((point_w * r2).sum()),
    }
    shares = {k: (v / total if total > 0 else float("nan")) for k, v in parts.items()}
    resid_full = np.full(np.shape(e_d), np.nan)
    resid_full[:, use] = resid
    return {
        "shares": shares,
        "ms_total": total / max(float(w.sum()), _EPS),
        "ms_parts": {k: v / max(float(w.sum()), _EPS) for k, v in parts.items()},
        "weighting": weighting,
        "offset": offset,  # [G]
        "jitter": jitter,  # [T, G]
        "group_mean": np.where(has, m, np.nan),  # [T, G]
        "counts": counts,  # [T, G]
        "residual": resid_full,  # [T, N]
    }


# --------------------------------------------------------------------------------------
# Camera-frame error
# --------------------------------------------------------------------------------------


def camera_frame_error(
    p_cross: np.ndarray,
    p_self_mapped: np.ndarray,
    pair_t: np.ndarray,
    pair_c: np.ndarray,
    min_points: int = 3,
) -> dict[str, Any]:
    """Gauge (best-fit Sim(3)) vs non-rigid split of cross-frame error, per ``(t, c)`` pair.

    ``p_self_mapped`` is the self-frame output ``P(n, s, t, t)`` mapped into camera ``c``
    by the GT relative pose; ``p_cross`` is ``P(n, s, t, c)``. Per pair the Sim(3)
    ``S: mapped -> cross`` is fitted; its rotation angle, ``|t_S| / median depth`` and
    ``log s`` are the pose error. ``nonrigid_share = sum SSE(cross, S(mapped)) /
    sum SSE(cross, mapped)`` with each pair's SSE divided by its mean depth squared, and
    ``gauge_share = 1 - nonrigid_share`` (the part a Sim(3) explains).
    """
    P = np.asarray(p_cross, dtype=np.float64)
    Q = np.asarray(p_self_mapped, dtype=np.float64)
    keys = np.stack([np.asarray(pair_t), np.asarray(pair_c)], axis=-1).astype(np.int64)
    ok = np.isfinite(P).all(-1) & np.isfinite(Q).all(-1)
    uniq, inv = np.unique(keys[ok], axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    idx_ok = np.flatnonzero(ok)
    rows = []
    for g in range(uniq.shape[0]):
        sel = idx_ok[inv == g]
        if sel.size < min_points:
            continue
        q, p = Q[sel], P[sel]
        s, R, t = geo.umeyama(q, p, with_scale=True)
        if not np.isfinite(s):
            continue
        fit = geo.apply_sim3(q, s, R, t)
        z_mean = float(np.mean(q[:, 2]))
        z_med = float(np.median(q[:, 2]))
        sse_tot = float(((p - q) ** 2).sum())
        sse_nr = float(((p - fit) ** 2).sum())
        rows.append(
            (
                uniq[g, 0], uniq[g, 1], sel.size, float(geo.rotation_angle_deg(R)),
                float(np.linalg.norm(t) / max(z_med, _EPS)), float(np.log(s)),
                np.sqrt(sse_tot / sel.size) / max(z_mean, _EPS),
                np.sqrt(sse_nr / sel.size) / max(z_mean, _EPS),
            )
        )
    arr = np.array(rows, dtype=np.float64).reshape(-1, 8)
    n_pts = arr[:, 2]
    tot2 = n_pts * arr[:, 6] ** 2
    nr2 = n_pts * arr[:, 7] ** 2
    nonrigid_share = float(nr2.sum() / tot2.sum()) if tot2.sum() > 0 else float("nan")
    return {
        "pair_t": arr[:, 0].astype(np.int64),
        "pair_c": arr[:, 1].astype(np.int64),
        "n_points": n_pts.astype(np.int64),
        "gauge_rot_deg": arr[:, 3],
        "gauge_trans_rel": arr[:, 4],
        "gauge_log_scale": arr[:, 5],
        "total_rms_rel": arr[:, 6],
        "nonrigid_rms_rel": arr[:, 7],
        "nonrigid_share": nonrigid_share,
        "gauge_share": 1.0 - nonrigid_share,
    }


# --------------------------------------------------------------------------------------
# Table entry point
# --------------------------------------------------------------------------------------


def _row_keys(d: dict[str, Any], tc: np.ndarray | None = None) -> np.ndarray:
    T = int(d["T"]) + 1
    tc = d["q_tc"] if tc is None else tc
    k = d["design"].astype(np.int64)
    k = k * (int(d["gt_xyz_cam"].shape[1]) + 2) + (d["q_track"].astype(np.int64) + 1)
    for a in (d["q_ts"], d["q_tt"], tc):
        k = k * T + np.asarray(a, dtype=np.int64)
    return k


def from_table(
    d: dict[str, Any],
    group_mode: str = "rigid",
    weighting: str = "points",
    self_design: int | str = "track",
    cross_designs: tuple[int | str, ...] = ("track", "cross"),
    **group_kwargs: Any,
) -> dict[str, Any]:
    """Run the A1 decomposition on one decoded table (see EXPERIMENTS.md section 3).

    Scale: global median scale over all self-frame rows with GT, design 5 excluded
    (:func:`table_scale`, reported as ``scale``). Depth offset/jitter use the self-frame rows of ``self_design`` (one per
    ``(track, t_t)``; duplicates are averaged). Camera-frame error uses cross-frame rows
    of ``cross_designs`` whose self-frame partner (same design, track, ``t_s``, ``t_t``) exists.
    """
    gt = select(d, has_gt=True)
    scale = table_scale(d)
    K = np.asarray(d["K_gt"], dtype=np.float64)
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    T, N = int(d["T"]), int(d["gt_xyz_cam"].shape[1])

    # Self-frame errors.
    sf = select(gt, design=self_design, self_frame=True)
    err = self_frame_errors(sf["p_xyz"] * scale, sf["g_xyz"], K)
    e_grid = np.zeros((T, N))
    cnt = np.zeros((T, N))
    ok = np.isfinite(err["e_d"])
    np.add.at(e_grid, (sf["q_tt"][ok], sf["q_track"][ok]), err["e_d"][ok])
    np.add.at(cnt, (sf["q_tt"][ok], sf["q_track"][ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_grid = np.where(cnt > 0, e_grid / cnt, np.nan)

    traj = world_trajectories(d["gt_xyz_cam"], T_w2c)
    z = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[..., 2]
    ref_depth = float(np.median(z[np.isfinite(z)]))
    group_kwargs.setdefault("ref_depth", ref_depth)
    labels = group_tracks(traj, d["is_static"], mode=group_mode, **group_kwargs)
    split = offset_jitter_split(e_grid, labels, weighting=weighting)

    sq = err["sq_err"]
    fin = np.isfinite(sq)
    sq_sum = float(sq[fin].sum())
    self_summary = {
        "n": int(fin.sum()),
        "sq_mean": sq_sum / max(int(fin.sum()), 1),
        "radial_share": float((err["radial"][fin] ** 2).sum() / sq_sum) if sq_sum > 0 else float("nan"),
        "perp_share": float((np.linalg.norm(err["perp"][fin], axis=-1) ** 2).sum() / sq_sum) if sq_sum > 0 else float("nan"),
        "e_d_rms": float(np.sqrt(np.nanmean(err["e_d"] ** 2))),
        "e_d_mean": float(np.nanmean(err["e_d"])),
        "angle_rms": float(np.sqrt(np.nanmean(err["angle"] ** 2))),
        "px_median": float(np.nanmedian(err["px"])),
        "px_mean": float(np.nanmean(err["px"])),
    }

    # Camera-frame error: cross rows matched to their self-frame partner.
    designs = [DESIGNS[x] if isinstance(x, str) else int(x) for x in cross_designs]
    cand = select(gt, mask=np.isin(gt["design"], designs))
    is_self = cand["q_tc"] == cand["q_tt"]
    self_keys = _row_keys(cand)[is_self]
    order = np.argsort(self_keys)
    self_keys = self_keys[order]
    self_rows = np.flatnonzero(is_self)[order]
    cross_rows = np.flatnonzero(~is_self)
    want = _row_keys(cand, tc=cand["q_tt"])[cross_rows]
    pos = np.clip(np.searchsorted(self_keys, want), 0, max(self_keys.size - 1, 0))
    found = self_keys.size > 0
    hit = (self_keys[pos] == want) if found else np.zeros(want.shape, dtype=bool)
    cam: dict[str, Any] = {"n_pairs_rows": int(hit.sum())}
    if hit.any():
        cr = cross_rows[hit]
        sr = self_rows[pos[hit]]
        t_idx = cand["q_tt"][cr].astype(np.int64)
        c_idx = cand["q_tc"][cr].astype(np.int64)
        T_rel = geo.relative_pose(T_w2c, t_idx, c_idx)
        mapped = geo.apply_transform(T_rel, cand["p_xyz"][sr] * scale)
        cam = camera_frame_error(cand["p_xyz"][cr] * scale, mapped, t_idx, c_idx)
        cam["n_pairs_rows"] = int(hit.sum())

    return {
        "scale": scale,
        "self": self_summary,
        "depth_split": split,
        "labels": labels,
        "camera_frame": cam,
        "e_d_grid": e_grid,
    }
