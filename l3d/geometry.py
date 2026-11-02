"""Geometry helpers for the L3D analysis code (numpy, float64, vectorized).

Conventions follow docs/l3d/EXPERIMENTS.md: right-handed cameras (+x right, +y down,
+z forward), ``T_w2c[t]`` maps world (= clip frame 0) to camera ``t``, normalized
pixels are ``u = x_px / (W - 1)``, ``v = y_px / (H - 1)``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_EPS = 1e-12


def _f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Rigid / similarity transforms
# --------------------------------------------------------------------------------------


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray | None = None,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted least-squares fit ``dst ~ s * R @ src + t`` (Umeyama 1991).

    Rows with non-finite values or non-positive weight are ignored. ``R`` is always a
    proper rotation (det = +1): when the best orthogonal map is a reflection the last
    singular direction is flipped, as in Umeyama's correction. With ``with_scale=False``
    the fit is SE(3) (``s = 1``). Returns NaNs when fewer than 3 points are usable or
    the source is degenerate.
    """
    src = _f64(src).reshape(-1, 3)
    dst = _f64(dst).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError(f"src {src.shape} and dst {dst.shape} differ")
    w = np.ones(src.shape[0], dtype=np.float64) if weights is None else _f64(weights).reshape(-1)
    valid = np.isfinite(src).all(-1) & np.isfinite(dst).all(-1) & np.isfinite(w) & (w > 0)
    nan_out = (float("nan"), np.full((3, 3), np.nan), np.full(3, np.nan))
    if valid.sum() < 3:
        return nan_out
    src, dst, w = src[valid], dst[valid], w[valid]
    w = w / w.sum()

    mu_src = w @ src
    mu_dst = w @ dst
    xs = src - mu_src
    xd = dst - mu_dst
    var_src = float(w @ (xs**2).sum(-1))
    if var_src <= _EPS:
        return nan_out
    cov = (xd * w[:, None]).T @ xs  # [3, 3], dst x src
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = float((sig * d).sum() / var_src) if with_scale else 1.0
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 matrix of ``x -> s R x + t``."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = float(s) * _f64(R)
    out[:3, 3] = _f64(t)
    return out


def apply_sim3(X: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``s R X + t`` for points ``X[..., 3]``."""
    return float(s) * (_f64(X) @ _f64(R).T) + _f64(t)


def apply_transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform(s) ``T[..., 4, 4]`` to points ``X[..., 3]`` (broadcast)."""
    T = _f64(T)
    X = _f64(X)
    return np.einsum("...ij,...j->...i", T[..., :3, :3], X) + T[..., :3, 3]


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transform(s) ``T[..., 4, 4]``."""
    T = _f64(T)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def nearest_rotation(R: np.ndarray) -> np.ndarray:
    """Closest proper rotation (Frobenius norm) to each ``R[..., 3, 3]``, via SVD."""
    u, _, vt = np.linalg.svd(_f64(R))
    d = np.sign(np.linalg.det(u @ vt))
    u = u.copy()
    u[..., :, 2] *= d[..., None]
    return u @ vt


def rigidify(T: np.ndarray) -> np.ndarray:
    """``T[..., 4, 4]`` with its rotation block replaced by :func:`nearest_rotation`.

    WorldTrack ``po_mini`` extrinsics are off from orthonormal by up to ~1e-3, enough to
    clip rotation angles to 0 and to distort GT mappings.
    """
    T = _f64(T).copy()
    T[..., :3, :3] = nearest_rotation(T[..., :3, :3])
    return T


def relative_pose(T_w2c: np.ndarray, t: np.ndarray | int, c: np.ndarray | int) -> np.ndarray:
    """``T_w2c[c] @ inv(T_w2c[t])``: maps camera-``t`` coordinates to camera ``c``.

    Rotations are projected to SO(3) first (:func:`rigidify`).
    """
    T_w2c = rigidify(T_w2c)
    t = np.asarray(t)
    c = np.asarray(c)
    return T_w2c[c] @ invert_se3(T_w2c[t])


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle (degrees) of rotation matrices ``R[..., 3, 3]``."""
    R = _f64(R)
    cos = 0.5 * (np.trace(R, axis1=-2, axis2=-1) - 1.0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vectors ``[..., 3]`` (radians) to matrices ``[..., 3, 3]``."""
    rotvec = _f64(rotvec)
    flat = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return flat.reshape(rotvec.shape[:-1] + (3, 3))


# --------------------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------------------


def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pinhole projection of camera-frame points ``X[..., 3]`` to pixels ``[..., 2]``.

    ``K`` is ``[3, 3]`` or broadcastable ``[..., 3, 3]``. Points with ``z <= 0`` give NaN.
    """
    K = _f64(K)
    X = _f64(X)
    x = np.einsum("...ij,...j->...i", K, X)
    z = x[..., 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = x[..., :2] / z
    return np.where(z > _EPS, uv, np.nan)


def unproject(K: np.ndarray, uv_px: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Camera-frame points with pixel ``uv_px[..., 2]`` and z-depth ``depth[...]``."""
    rays = ray_dirs(K, uv_px, normalize=False)
    return rays * _f64(depth)[..., None]


def uv_norm_to_px(uv: np.ndarray, hw: tuple[int, int] | np.ndarray) -> np.ndarray:
    """Normalized ``[0, 1]`` pixels to pixels, with the ``(W - 1), (H - 1)`` convention."""
    h, w = int(hw[0]), int(hw[1])
    scale = np.array([max(w - 1, 1), max(h - 1, 1)], dtype=np.float64)
    return _f64(uv) * scale


def uv_px_to_norm(uv_px: np.ndarray, hw: tuple[int, int] | np.ndarray) -> np.ndarray:
    """Inverse of :func:`uv_norm_to_px`."""
    h, w = int(hw[0]), int(hw[1])
    scale = np.array([max(w - 1, 1), max(h - 1, 1)], dtype=np.float64)
    return _f64(uv_px) / scale


def ray_dirs(K: np.ndarray, uv_px: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Camera-frame ray directions ``K^-1 [u; v; 1]`` (unit length if ``normalize``)."""
    K = _f64(K)
    uv_px = _f64(uv_px)
    homo = np.concatenate([uv_px, np.ones(uv_px.shape[:-1] + (1,))], axis=-1)
    d = np.einsum("...ij,...j->...i", np.linalg.inv(K), homo)
    if normalize:
        d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), _EPS)
    return d


def ls_ray_center(
    points: np.ndarray,
    dirs: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point closest (least squares) to lines ``p_i + lambda d_i``, plus per-line distances.

    Minimizes ``sum_i w_i |(I - d_i d_i^T)(o - p_i)|^2`` in closed form (one 3x3 solve
    per batch). ``points``/``dirs`` are ``[..., R, 3]``; leading dims are batched. Lines
    with non-finite values get zero weight; their distance is still reported (NaN).
    Returns ``(center[..., 3], dist[..., R])``.
    """
    p = _f64(points)
    d = _f64(dirs)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), _EPS)
    w = np.ones(p.shape[:-1], dtype=np.float64) if weights is None else _f64(weights) * np.ones(p.shape[:-1])
    finite = np.isfinite(p).all(-1) & np.isfinite(d).all(-1) & np.isfinite(w)
    w = np.where(finite, w, 0.0)
    p0 = np.where(finite[..., None], p, 0.0)
    d0 = np.where(finite[..., None], d, 0.0)

    proj = np.eye(3) - d0[..., :, None] * d0[..., None, :]  # [..., R, 3, 3]
    A = np.einsum("...r,...rij->...ij", w, proj)
    b = np.einsum("...r,...rij,...rj->...i", w, proj, p0)
    center = np.einsum("...ij,...j->...i", np.linalg.pinv(A, hermitian=True), b)

    diff = center[..., None, :] - p
    perp = diff - (diff * d).sum(-1, keepdims=True) * d
    return center, np.linalg.norm(perp, axis=-1)


def fit_intrinsics_from_points(
    xyz_cam: np.ndarray,
    uv_px: np.ndarray,
    hw: tuple[int, int] | np.ndarray,
) -> np.ndarray:
    """Pinhole ``K`` from camera-frame points and their pixels.

    Same estimator as ``src/eval/tasks.py::_estimate_intrinsics_params_from_predictions``:
    principal point at the image center ``((W - 1) / 2, (H - 1) / 2)`` and median
    per-point ``fx = z |u - cx| / |x|`` (likewise ``fy``), but taking pixels, not
    normalized coordinates.
    """
    h, w = int(hw[0]), int(hw[1])
    cx = 0.5 * float(max(w - 1, 1))
    cy = 0.5 * float(max(h - 1, 1))
    pts = _f64(xyz_cam).reshape(-1, 3)
    uv = _f64(uv_px).reshape(-1, 2)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_vals = z * np.abs(uv[:, 0] - cx) / np.maximum(np.abs(x), 1e-6)
        fy_vals = z * np.abs(uv[:, 1] - cy) / np.maximum(np.abs(y), 1e-6)
    fx_vals = fx_vals[np.isfinite(fx_vals) & (fx_vals > 1e-6)]
    fy_vals = fy_vals[np.isfinite(fy_vals) & (fy_vals > 1e-6)]
    fx = float(np.median(fx_vals)) if fx_vals.size > 0 else float(max(w, 1))
    fy = float(np.median(fy_vals)) if fy_vals.size > 0 else float(max(h, 1))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def K_from_fxfycxcy(params: np.ndarray) -> np.ndarray:
    """``[fx, fy, cx, cy]`` to a 3x3 intrinsics matrix."""
    fx, fy, cx, cy = (float(v) for v in np.asarray(params).reshape(4))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------------------


def median_scale(gt: np.ndarray, pred: np.ndarray) -> float:
    """``median |gt| / median |pred|`` over finite pairs.

    Copy of ``_compute_scale_factor_global`` in ``eval_track3d_in_worldtrack.py``.
    """
    gt_flat = _f64(gt).reshape(-1, 3)
    pred_flat = _f64(pred).reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if not np.any(finite):
        return 1.0
    gt_norm = np.maximum(np.linalg.norm(gt_flat[finite], axis=-1), _EPS)
    pred_norm = np.maximum(np.linalg.norm(pred_flat[finite], axis=-1), _EPS)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), _EPS))
