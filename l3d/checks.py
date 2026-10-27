"""The five consistency residuals of the proposal ("Checks") and A4 pose helpers.

Every 3D residual is divided by the mean depth (z) of the points involved, so shrinking
the scene cannot lower it. Each check returns per-query (or per-point) residuals so it
can be scored as a detector (``l3d.detection``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from l3d import geometry as geo

_EPS = 1e-12


def _f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def robust_psi(r: np.ndarray, kind: str | None = None, scale: float = 1.0) -> np.ndarray:
    """Robust penalty on non-negative residuals: ``None`` (identity), ``huber``, ``cauchy``."""
    r = _f64(r)
    if kind is None or kind == "none":
        return r
    c = float(scale)
    if kind == "huber":
        return np.where(r <= c, 0.5 * r**2, c * (r - 0.5 * c))
    if kind == "cauchy":
        return 0.5 * c**2 * np.log1p((r / c) ** 2)
    raise ValueError(f"unknown psi '{kind}'")


def group_index(*keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of stacked integer keys and the inverse index (for per-pair fits)."""
    stacked = np.stack([np.asarray(k, dtype=np.int64) for k in keys], axis=-1)
    uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)


# --------------------------------------------------------------------------------------
# (1) Source swap
# --------------------------------------------------------------------------------------


def source_swap(
    P_a: np.ndarray,
    P_b: np.ndarray,
    psi: str | None = None,
    psi_scale: float = 1.0,
) -> np.ndarray:
    """``psi(|P_a - P_b| / zbar)`` with ``zbar`` the pair's mean depth.

    ``P_a = P(u1, t1, t3, c)``, ``P_b = P(u2, t2, t3, c)``, both ``[M, 3]`` in camera ``c``.
    """
    P_a = _f64(P_a)
    P_b = _f64(P_b)
    zbar = 0.5 * (P_a[..., 2] + P_b[..., 2])
    r = np.linalg.norm(P_a - P_b, axis=-1) / np.maximum(np.abs(zbar), _EPS)
    return robust_psi(r, psi, psi_scale)


# --------------------------------------------------------------------------------------
# (2) Camera swap
# --------------------------------------------------------------------------------------


def camera_swap(
    P_cam_a: np.ndarray,
    P_cam_b: np.ndarray,
    with_scale: bool = False,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Umeyama fit ``T*: a -> b`` (SE(3), or Sim(3) with ``with_scale``) and its residual.

    ``P_cam_a``/``P_cam_b`` are the same pixels decoded in cameras ``a`` and ``b``
    (``[M, 3]``). Per-point residual ``|T* P_a - P_b| / zbar``, ``zbar`` the mean of the
    point's depth in ``a`` and in ``b``. Returns ``residual``, ``s``, ``R``, ``t``, ``T`` (4x4).
    """
    P_a = _f64(P_cam_a)
    P_b = _f64(P_cam_b)
    s, R, t = geo.umeyama(P_a, P_b, weights=weights, with_scale=with_scale)
    fit = geo.apply_sim3(P_a, s, R, t)
    zbar = 0.5 * (P_a[:, 2] + P_b[:, 2])
    res = np.linalg.norm(fit - P_b, axis=-1) / np.maximum(np.abs(zbar), _EPS)
    return {"residual": res, "s": s, "R": R, "t": t, "T": geo.sim3_matrix(s, R, t)}


# --------------------------------------------------------------------------------------
# (3) Ray-center agreement
# --------------------------------------------------------------------------------------


def ray_center(
    uv_px_t: np.ndarray,
    P_in_c: np.ndarray,
    K: np.ndarray,
    R_star: np.ndarray,
    t_star: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Lines through ``P_in_c`` with directions ``R* K^-1 [u; 1]`` should meet at camera t's center.

    ``uv_px_t`` are frame-t query pixels ``[M, 2]``, ``P_in_c`` their points decoded in
    camera ``c`` (``t_c != t_t``). Returns the least-squares ``center``, per-ray
    perpendicular distance / mean depth (``spread``) and, if ``t_star`` (camera t's
    center in camera c, i.e. the translation of ``T*: t -> c``) is given,
    ``center_dist = |center - t*| / median depth``. Depth = z of ``P_in_c``. Pass GT
    ``K`` and rotation (and translation) for the GT variant.
    """
    P = _f64(P_in_c)
    dirs = geo.ray_dirs(K, uv_px_t) @ _f64(R_star).T
    center, dist = geo.ls_ray_center(P, dirs, weights)
    z = P[:, 2][np.isfinite(P[:, 2])]
    z_mean = float(np.mean(z)) if z.size else float("nan")
    z_med = float(np.median(z)) if z.size else float("nan")
    out: dict[str, Any] = {"center": center, "spread": dist / max(abs(z_mean), _EPS)}
    if t_star is not None:
        out["center_dist"] = float(np.linalg.norm(center - _f64(t_star)) / max(abs(z_med), _EPS))
    return out


# --------------------------------------------------------------------------------------
# (4) Reprojection, (5) static rigidity (analysis only)
# --------------------------------------------------------------------------------------


def reprojection(P_self: np.ndarray, K: np.ndarray, uv_ref_px: np.ndarray) -> np.ndarray:
    """Pixel distance between ``K``-projected self-frame outputs and reference pixels."""
    return np.linalg.norm(geo.project(K, P_self) - _f64(uv_ref_px), axis=-1)


def static_rigidity(P: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Static points decoded at several ``t_t`` in one camera ``c`` must not move.

    ``P`` is ``[n, k, 3]`` (``k`` target frames); ``mask`` ``[n, k]`` marks valid entries.
    Returns ``residual [n, k]`` = distance to the point's mean / its mean depth, and
    ``per_point [n]`` = RMS over ``k``.
    """
    P = _f64(P)
    m = np.isfinite(P).all(-1) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(P).all(-1))
    cnt = m.sum(-1).astype(np.float64)
    P0 = np.where(m[..., None], P, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = P0.sum(1) / cnt[:, None]
        res = np.linalg.norm(P - mean[:, None], axis=-1) / np.abs(mean[:, None, 2])
    res = np.where(m, res, np.nan)
    with np.errstate(invalid="ignore"):
        per_point = np.sqrt(np.nanmean(res**2, axis=1)) if res.size else np.zeros(0)
    return {"residual": res, "per_point": per_point}


# --------------------------------------------------------------------------------------
# A4: pose from a decoded grid, pose errors
# --------------------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_f64(x)))


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return geo.sim3_matrix(1.0, R, t)


def pose_from_grid(
    P_self: np.ndarray,
    P_cross: np.ndarray,
    conf: np.ndarray | None = None,
    uv_px: np.ndarray | None = None,
    K: np.ndarray | None = None,
    with_scale: bool = False,
) -> dict[str, dict[str, Any]]:
    """Relative pose ``T_{t->c}`` from frame-t pixels decoded in camera t and in camera c.

    Methods: ``umeyama``; ``umeyama_conf`` (weights ``sigmoid(conf)``, if ``conf`` given);
    ``ray_center`` (rotation of the (weighted) Umeyama fit, translation = ray center of
    lines through ``P_cross`` with directions ``R K^-1 [u; 1]``; needs ``uv_px`` and ``K``).
    Each entry has ``T`` (4x4 rigid), ``s`` (fit scale, 1 for SE(3)).
    """
    out: dict[str, dict[str, Any]] = {}
    s, R, t = geo.umeyama(P_self, P_cross, with_scale=with_scale)
    out["umeyama"] = {"T": _se3(R, t), "s": s}
    w = None
    if conf is not None:
        w = _sigmoid(conf)
        s_w, R_w, t_w = geo.umeyama(P_self, P_cross, weights=w, with_scale=with_scale)
        out["umeyama_conf"] = {"T": _se3(R_w, t_w), "s": s_w}
    if uv_px is not None and K is not None:
        R_rc = out["umeyama_conf"]["T"][:3, :3] if w is not None else R
        rc = ray_center(uv_px, P_cross, K, R_rc, weights=w)
        out["ray_center"] = {"T": _se3(R_rc, rc["center"]), "s": 1.0}
    return out


def pose_errors(T_pred: np.ndarray, T_gt: np.ndarray, scale: float = 1.0) -> dict[str, np.ndarray]:
    """Rotation error (deg) and translation error ``|scale * t_pred - t_gt|`` of ``[..., 4, 4]``.

    ``scale`` is the clip's scale alignment (1 if the outputs were already scaled).
    """
    T_pred = _f64(T_pred)
    T_gt = _f64(T_gt)
    R_err = np.swapaxes(T_pred[..., :3, :3], -1, -2) @ T_gt[..., :3, :3]
    t_err = np.linalg.norm(float(scale) * T_pred[..., :3, 3] - T_gt[..., :3, 3], axis=-1)
    return {"rot_deg": geo.rotation_angle_deg(R_err), "trans": t_err}


def ate_rpe(
    c2w_pred: np.ndarray,
    c2w_gt: np.ndarray,
    align: str = "sim3",
    delta: int = 1,
) -> dict[str, float]:
    """ATE (RMS camera-center error after alignment) and RPE over a clip trajectory.

    ``align``: ``"sim3"`` (Umeyama with scale on camera centers), ``"se3"`` or ``"none"``.
    RPE compares relative motions over ``delta`` frames (translations scaled by the
    alignment scale): RMS translation error and mean rotation error (deg).
    """
    Pp = _f64(c2w_pred)
    Pg = _f64(c2w_gt)
    cp, cg = Pp[:, :3, 3], Pg[:, :3, 3]
    if align == "none":
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, R, t = geo.umeyama(cp, cg, with_scale=(align == "sim3"))
        if not np.isfinite(s):  # e.g. a static camera: fall back to translation only
            s, R, t = 1.0, np.eye(3), (cg - cp).mean(0)
    ca = geo.apply_sim3(cp, s, R, t)
    ate = float(np.sqrt(np.mean(((ca - cg) ** 2).sum(-1))))
    rel_p = geo.invert_se3(Pp[:-delta]) @ Pp[delta:]
    rel_g = geo.invert_se3(Pg[:-delta]) @ Pg[delta:]
    rel_p = rel_p.copy()
    rel_p[:, :3, 3] *= s
    err = geo.invert_se3(rel_g) @ rel_p
    rpe_t = float(np.sqrt(np.mean((err[:, :3, 3] ** 2).sum(-1)))) if err.size else float("nan")
    rpe_r = float(np.mean(geo.rotation_angle_deg(err[:, :3, :3]))) if err.size else float("nan")
    return {"ate": ate, "rpe_trans": rpe_t, "rpe_rot_deg": rpe_r, "align_scale": float(s)}
