"""GAUGE baseline (Feng et al., arXiv:2609.32596): group-wise ray-scale correction from GT anchors.

A reference, not label-free. Per co-moving group with anchors (GT tracks of a fraction
of the points): a per-frame radial scale ``alpha_t`` about camera ``t``'s center, the
median of ``|Q - C_t| / |P - C_t|`` over the anchors seen at ``t`` (frames without
anchors linearly interpolated, Gaussian smoothed with sigma 2, else the group-level
scale about ``C_0``), then one median translation ``delta``; static (world-fixed)
groups scale about their first anchor instead. Points outside an anchored group keep
the baseline. Deviations from the paper: groups are the GT motion groups of A1 (the
paper's "oracle" variant) instead of its unsupervised grouping, anchors are whole
tracks (GT wherever both GT and predicted visibility hold), and camera centers come
from the model's own grid poses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from l3d import geometry as geo

SMOOTH_SIGMA = 2.0


def allocate_anchors(
    labels: np.ndarray, motion: np.ndarray, candidates: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Anchor mask ``[N]``: ``floor(frac * N)`` anchors split over groups by
    ``n_g (1 + m_g / max m)`` with largest-remainder rounding, capped by each group's candidates."""
    groups = np.unique(labels[labels >= 0])
    n_total = int(np.floor(frac * np.count_nonzero(labels >= 0)))
    anchors = np.zeros(labels.shape, dtype=bool)
    if n_total == 0 or groups.size == 0:
        return anchors
    n_g = np.array([np.count_nonzero(labels == g) for g in groups], dtype=np.float64)
    m_g = np.array([np.nanmean(motion[labels == g]) if np.any(np.isfinite(motion[labels == g])) else 0.0 for g in groups])
    m_max = m_g.max() if m_g.max() > 0 else 1.0
    score = n_g * (1.0 + m_g / m_max)
    quota = n_total * score / score.sum()
    k = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - k))[: n_total - k.sum()]:
        k[i] += 1
    for g, kg in zip(groups, k):
        pool = np.flatnonzero((labels == g) & candidates)
        if kg > 0 and pool.size:
            anchors[rng.choice(pool, size=min(kg, pool.size), replace=False)] = True
    return anchors


def _ratio(Q: np.ndarray, P: np.ndarray, C: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.linalg.norm(Q - C, axis=-1) / np.linalg.norm(P - C, axis=-1)


def _frame_scales(Q: np.ndarray, P: np.ndarray, known: np.ndarray, centers: np.ndarray, fallback: float) -> np.ndarray:
    """Per-frame median ratio about ``centers[t]`` (``Q, P, known``: ``[T, A, ...]``), interpolated and smoothed."""
    T = P.shape[0]
    alpha = np.full(T, np.nan)
    for t in range(T):
        r = _ratio(Q[t][known[t]], P[t][known[t]], centers[t])
        r = r[np.isfinite(r) & (r > 0)]
        if r.size:
            alpha[t] = np.median(r)
    have = np.isfinite(alpha)
    if not have.any():
        return np.full(T, fallback)
    alpha = np.interp(np.arange(T), np.flatnonzero(have), alpha[have])
    return gaussian_filter1d(alpha, SMOOTH_SIGMA, mode="nearest")


def gauge_correct(
    P: np.ndarray, Q: np.ndarray, known: np.ndarray, labels: np.ndarray, anchors: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """GAUGE-corrected copy of ``P [T, N, 3]`` (world = camera 0, model scale already applied).

    ``Q`` GT ``[T, N, 3]``; ``known [T, N]`` where an anchor's GT may be used; ``labels [N]``
    (``0`` static, ``>= 1`` dynamic groups, ``-1`` none); ``centers [T, 3]`` camera centers.
    """
    out = P.copy()
    for g in np.unique(labels[labels >= 0]):
        members = labels == g
        a = members & anchors
        if not a.any():
            continue
        Pa, Qa, ka = P[:, a], Q[:, a], known[:, a]
        if g == 0:  # world-fixed: one scale about the first anchor's mean position
            first = np.nanmean(np.where(ka[..., None], Pa, np.nan)[:, 0], axis=0)
            first_q = np.nanmean(np.where(ka[..., None], Qa, np.nan)[:, 0], axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.linalg.norm(Qa[ka] - first_q, axis=-1) / np.linalg.norm(Pa[ka] - first, axis=-1)
            r = r[np.isfinite(r) & (r > 0)]
            alpha = float(np.median(r)) if r.size else 1.0
            scaled = first + alpha * (P[:, members] - first)
            scaled_a = first + alpha * (Pa - first)
        else:
            r0 = _ratio(Qa[ka], Pa[ka], centers[0])
            r0 = r0[np.isfinite(r0) & (r0 > 0)]
            alpha_g = float(np.median(r0)) if r0.size else 1.0
            alpha = _frame_scales(Qa, Pa, ka, centers, alpha_g)
            scaled = centers[:, None] + alpha[:, None, None] * (P[:, members] - centers[:, None])
            scaled_a = centers[:, None] + alpha[:, None, None] * (Pa - centers[:, None])
        d = (Qa - scaled_a)[ka]
        delta = np.median(d[np.isfinite(d).all(-1)], axis=0) if np.isfinite(d).all(-1).any() else np.zeros(3)
        out[:, members] = scaled + delta
    return out


def sim3_per_group(P: np.ndarray, Q: np.ndarray, known: np.ndarray, labels: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Reference: per anchored group, one Sim(3) (Umeyama) from its anchors' predictions to GT."""
    out = P.copy()
    for g in np.unique(labels[labels >= 0]):
        members = labels == g
        a = members & anchors
        ka = known[:, a]
        if ka.sum() < 3:
            continue
        s, R, t = geo.umeyama(P[:, a][ka], Q[:, a][ka], with_scale=True)
        if np.isfinite(s):
            out[:, members] = geo.apply_sim3(P[:, members], s, R, t)
    return out


THRESHOLDS = (0.1, 0.3, 0.5, 1.0)  # upstream PIXEL_TO_FIXED_METRIC_THRESH


def apd_epe(P: np.ndarray, Q: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Upstream APD (mean share within the fixed thresholds) and EPE over ``mask [T, N]`` entries, no re-alignment."""
    dist = np.linalg.norm(P - Q, axis=-1)[mask]
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return {"apd": float("nan"), "epe": float("nan"), "n": 0}
    return {"apd": float(np.mean([np.mean(dist <= th) for th in THRESHOLDS])), "epe": float(dist.mean()), "n": int(dist.size)}
