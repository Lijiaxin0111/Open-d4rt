"""Per-row check residuals on a decoded table (A3 detection) and pose baselines (A4).

Every function takes a full table (:func:`l3d.table.load_table`) and the per-row errors of
:func:`row_errors`, which multiplies the outputs by the clip scale first. Errors are
relative, ``rel = |s P - X| / |X|``; confidences are ``sigmoid(conf_logit)``. Each
``*_rows`` function returns flat per-row arrays: residual(s), ``error`` (total), ``target``
(the component the check is predicted to see, Table 1) and ``conf``, all of one length.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from l3d import checks
from l3d import geometry as geo
from l3d.table import DESIGNS, source_swap_rows

_EPS = 1e-12
MIN_POINTS = 8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def row_errors(d: dict[str, Any], scale: float) -> dict[str, np.ndarray]:
    """Scaled outputs ``P`` and per-row ``rel``, ``e_d`` (log depth ratio along the GT ray),
    ``angle`` (image-plane error / ``|X|``) and ``conf``; NaN where a row has no GT."""
    P = d["p_xyz"].astype(np.float64) * scale
    X = d["g_xyz"].astype(np.float64)
    dist = np.linalg.norm(X, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ray = X / dist[:, None]
        along = (P * ray).sum(-1)
        e_d = np.where(along > _EPS, np.log(np.maximum(along, _EPS) / dist), np.nan)
        perp = P - along[:, None] * ray
        rel = np.linalg.norm(P - X, axis=-1) / dist
        angle = np.linalg.norm(perp, axis=-1) / dist
    return {"P": P, "rel": rel, "e_d": e_d, "angle": angle, "conf": _sigmoid(d["p_conf_logit"])}


def uv_px(d: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """Native-pixel source pixels of ``rows``."""
    return geo.uv_norm_to_px(np.stack([d["q_u"][rows], d["q_v"][rows]], -1), d["hw_native"])


def _pixel_ident(d: dict[str, Any], rows: np.ndarray, design: str) -> np.ndarray:
    """Track for design 1; for the grid (no tracks) an id per distinct source pixel."""
    if design != "grid":
        return d["q_track"][rows].astype(np.int64)
    uv = np.stack([d["q_u"][rows], d["q_v"][rows]], -1)
    return np.unique(uv, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)


def cross_pairs(d: dict[str, Any], design: str = "cross") -> tuple[np.ndarray, np.ndarray]:
    """Row pairs ``(j, i)`` of one design: ``j`` decoded in camera ``t_t``, ``i`` the same query
    (track or grid pixel, ``t_s``, ``t_t``) decoded in another camera."""
    rows = np.flatnonzero(d["design"] == DESIGNS[design])
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    T = int(d["T"])
    ident = _pixel_ident(d, rows, design)
    s, t, c = (d[k][rows].astype(np.int64) for k in ("q_ts", "q_tt", "q_tc"))
    key = ((ident + 1) * T + s) * T + t
    is_self = c == t
    order = np.argsort(key[is_self], kind="stable")
    self_key, self_rows = key[is_self][order], rows[is_self][order]
    want, cross = key[~is_self], rows[~is_self]
    if self_key.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos = np.clip(np.searchsorted(self_key, want), 0, self_key.size - 1)
    hit = self_key[pos] == want
    return self_rows[pos[hit]], cross[hit]


def _groups(*keys: np.ndarray) -> list[np.ndarray]:
    if keys[0].size == 0:
        return []
    _, inv = checks.group_index(*keys)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def self_rows(d: dict[str, Any], design: str = "cross") -> np.ndarray:
    """Rows with ``t_s == t_t == t_c`` (their source pixel is the point's pixel at ``t_t``)."""
    return np.flatnonzero((d["design"] == DESIGNS[design]) & (d["q_ts"] == d["q_tt"]) & (d["q_tc"] == d["q_tt"]))


def fit_K(d: dict[str, Any], err: dict[str, np.ndarray], design: str = "cross") -> np.ndarray:
    """Label-free intrinsics from the self-frame outputs of ``design`` and their query pixels."""
    r = self_rows(d, design)
    return geo.fit_intrinsics_from_points(err["P"][r], uv_px(d, r), d["hw_native"])


def _empty(*names: str) -> dict[str, np.ndarray]:
    return {k: np.zeros(0) for k in names}


# --------------------------------------------------------------------------------------
# A3 detection rows
# --------------------------------------------------------------------------------------


def source_swap_detection(d: dict[str, Any], err: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Source swap with the model's ``u2`` (design 5) and the GT-``u2`` control (design 1).

    Target: the source dependence of depth error, ``|e_d(u1, t1) - e_d(u2_gt, t2)|`` at the
    same ``(t3, c)``. Error and confidence: mean over the pair.
    """
    rows = source_swap_rows(d)
    a, b, g = rows["a"], rows["b"], rows["g"]
    P, rel, conf = err["P"], err["rel"], err["conf"]
    target = np.abs(err["e_d"][a] - err["e_d"][g])
    return {
        "model_u2": {
            "residual": checks.source_swap(P[a], P[b]), "error": 0.5 * (rel[a] + rel[b]),
            "target": target, "conf": 0.5 * (conf[a] + conf[b]),
        },
        "gt_u2": {
            "residual": checks.source_swap(P[a], P[g]), "error": 0.5 * (rel[a] + rel[g]),
            "target": target, "conf": 0.5 * (conf[a] + conf[g]),
        },
    }


def camera_swap_detection(d: dict[str, Any], err: dict[str, np.ndarray], min_points: int = MIN_POINTS) -> dict[str, np.ndarray]:
    """Camera swap per ``(t_s, t_t, c)`` group of design 1: the same tracks decoded in camera
    ``t_t`` and in camera ``c``, SE(3) and Sim(3) Umeyama residuals.

    Target: non-rigid camera-frame error, the residual of the cross output after the best
    Sim(3) fit from the self-frame output mapped by the GT pose (as in A1), over ``|X|``.
    """
    j, i = cross_pairs(d, "cross")
    P = err["P"]
    ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1)
    j, i = j[ok], i[ok]
    if i.size == 0:
        return _empty("residual_se3", "residual_sim3", "error", "target", "conf")
    res_se3, res_sim3, target = (np.full(i.size, np.nan) for _ in range(3))
    T_w2c = d["T_w2c_gt"]
    X = d["g_xyz"].astype(np.float64)
    for sel in _groups(d["q_ts"][i], d["q_tt"][i], d["q_tc"][i]):
        if sel.size < min_points:
            continue
        a, b = P[j[sel]], P[i[sel]]
        res_se3[sel] = checks.camera_swap(a, b)["residual"]
        res_sim3[sel] = checks.camera_swap(a, b, with_scale=True)["residual"]
        T_rel = geo.relative_pose(T_w2c, int(d["q_tt"][i[sel[0]]]), int(d["q_tc"][i[sel[0]]]))
        mapped = geo.apply_transform(T_rel, a)
        s, R, t = geo.umeyama(mapped, b, with_scale=True)
        if np.isfinite(s):
            target[sel] = np.linalg.norm(b - geo.apply_sim3(mapped, s, R, t), axis=-1) / np.linalg.norm(X[i[sel]], axis=-1)
    keep = np.isfinite(res_se3)
    return {
        "residual_se3": res_se3[keep], "residual_sim3": res_sim3[keep], "error": err["rel"][i][keep],
        "target": target[keep], "conf": err["conf"][i][keep],
    }


def ray_center_detection(
    d: dict[str, Any], err: dict[str, np.ndarray], K_fit: np.ndarray, min_points: int = MIN_POINTS
) -> dict[str, np.ndarray]:
    """Ray-center spread per ``(t, c)`` group of design-1 rows with ``t_s = t_t = t``.

    Label-free: ``K_fit`` and the SE(3) camera-swap rotation; GT variant: ``K_gt`` and GT pose.
    Target: image-plane error of the self-frame output ``P(u, t, t, t)``. Error and
    confidence: the cross row.
    """
    j, i = cross_pairs(d, "cross")
    keep = d["q_ts"][i] == d["q_tt"][i]
    j, i = j[keep], i[keep]
    P = err["P"]
    ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1)
    j, i = j[ok], i[ok]
    if i.size == 0:
        return _empty("residual", "residual_gt", "error", "target", "conf")
    spread, spread_gt = np.full(i.size, np.nan), np.full(i.size, np.nan)
    K_gt = np.asarray(d["K_gt"], dtype=np.float64)
    uv = uv_px(d, i)
    for sel in _groups(d["q_tt"][i], d["q_tc"][i]):
        if sel.size < min_points:
            continue
        a, b = P[j[sel]], P[i[sel]]
        cs = checks.camera_swap(a, b)
        if not np.isfinite(cs["R"]).all():
            continue
        spread[sel] = checks.ray_center(uv[sel], b, K_fit, cs["R"], cs["t"])["spread"]
        T_rel = geo.relative_pose(d["T_w2c_gt"], int(d["q_tt"][i[sel[0]]]), int(d["q_tc"][i[sel[0]]]))
        spread_gt[sel] = checks.ray_center(uv[sel], b, K_gt, T_rel[:3, :3], T_rel[:3, 3])["spread"]
    keep = np.isfinite(spread)
    return {
        "residual": spread[keep], "residual_gt": spread_gt[keep], "error": err["rel"][i][keep],
        "target": err["angle"][j][keep], "conf": err["conf"][i][keep],
    }


def reprojection_detection(d: dict[str, Any], err: dict[str, np.ndarray], K_fit: np.ndarray) -> dict[str, np.ndarray]:
    """Reprojection (px) of design-1 self-frame outputs with ``t_s = t_t`` against their query
    pixel, with ``K_fit``. Target: image-plane error of the same row."""
    r = self_rows(d, "cross")
    r = r[np.isfinite(err["P"][r]).all(-1)]
    px = checks.reprojection(err["P"][r], K_fit, uv_px(d, r))
    return {"residual": px, "error": err["rel"][r], "target": err["angle"][r], "conf": err["conf"][r]}


def static_rigidity_detection(d: dict[str, Any], err: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Static rigidity of design 3 (GT-static tracks decoded at every ``t_t`` in camera ``c``):
    distance to the ``(track, c)`` mean / its depth. Target: depth error ``|e_d|``."""
    r = np.flatnonzero(d["design"] == DESIGNS["static"])
    r = r[np.isfinite(err["P"][r]).all(-1)]
    if r.size == 0:
        return _empty("residual", "error", "target", "conf")
    _, inv = checks.group_index(d["q_track"][r], d["q_tc"][r])
    cnt = np.bincount(inv).astype(np.float64)
    mean = np.stack([np.bincount(inv, weights=err["P"][r, k]) for k in range(3)], -1) / cnt[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.linalg.norm(err["P"][r] - mean[inv], axis=-1) / np.abs(mean[inv, 2])
    keep = (cnt[inv] >= 2) & np.isfinite(res)
    return {"residual": res[keep], "error": err["rel"][r][keep], "target": np.abs(err["e_d"][r][keep]),
            "conf": err["conf"][r][keep]}


# --------------------------------------------------------------------------------------
# A4 pose baselines
# --------------------------------------------------------------------------------------


POSE_METHODS = ("umeyama", "umeyama_conf", "ray_center", "ray_center_gtK")


def pose_baselines(d: dict[str, Any], err: dict[str, np.ndarray], min_points: int = MIN_POINTS) -> dict[str, Any]:
    """Relative pose ``T_{t->c}`` per design-2 ``(t, c)`` pair by each of :data:`POSE_METHODS`.

    ``umeyama_conf`` weights each pixel by ``sigmoid(conf_t) * sigmoid(conf_c)``; ray center
    uses its rotation with label-free ``K`` (fitted on the grid's self-frame outputs) or
    ``K_gt``. Translations are relative to the clip's median GT depth. Also returns per-pair
    medians of the label-free camera-swap (SE(3)) and ray-center residuals, and ATE / RPE of
    the trajectory ``T_{t->0}`` (frame 0 = identity) per method.
    """
    j, i = cross_pairs(d, "grid")
    P = err["P"]
    ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1)
    j, i = j[ok], i[ok]
    K_fit = fit_K(d, err, "grid")
    K_gt = np.asarray(d["K_gt"], dtype=np.float64)
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    z = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[..., 2]
    ref_depth = float(np.median(z[np.isfinite(z)]))
    T = int(d["T"])
    uv = uv_px(d, i)
    w_logit = _logit(err["conf"][j] * err["conf"][i])
    pairs: dict[str, list[float]] = {k: [] for k in ("t", "c", "gt_rot_deg", "gt_trans_rel", "cs_residual", "rc_spread")}
    per_method = {m: {"rot_deg": [], "trans_rel": []} for m in POSE_METHODS}
    traj = {m: np.tile(np.eye(4), (T, 1, 1)) for m in POSE_METHODS}
    have = np.zeros(T, dtype=bool)
    have[0] = True
    for sel in _groups(d["q_tt"][i], d["q_tc"][i]):
        if sel.size < min_points:
            continue
        t, c = int(d["q_tt"][i[sel[0]]]), int(d["q_tc"][i[sel[0]]])
        a, b = P[j[sel]], P[i[sel]]
        est = checks.pose_from_grid(a, b, conf=w_logit[sel], uv_px=uv[sel], K=K_fit)
        if not all(np.isfinite(e["T"]).all() for e in est.values()):
            continue
        R_rc = est["umeyama_conf"]["T"][:3, :3]
        rc_gt = checks.ray_center(uv[sel], b, K_gt, R_rc, weights=_sigmoid(w_logit[sel]))
        est["ray_center_gtK"] = {"T": geo.sim3_matrix(1.0, R_rc, rc_gt["center"]), "s": 1.0}
        T_gt = geo.relative_pose(T_w2c, t, c)
        cs = checks.camera_swap(a, b)
        rc = checks.ray_center(uv[sel], b, K_fit, cs["R"], cs["t"])
        pairs["t"].append(t)
        pairs["c"].append(c)
        pairs["gt_rot_deg"].append(float(geo.rotation_angle_deg(T_gt[:3, :3])))
        pairs["gt_trans_rel"].append(float(np.linalg.norm(T_gt[:3, 3]) / ref_depth))
        pairs["cs_residual"].append(float(np.median(cs["residual"])))
        pairs["rc_spread"].append(float(np.median(rc["spread"])))
        for m in POSE_METHODS:
            pe = checks.pose_errors(est[m]["T"], T_gt)
            per_method[m]["rot_deg"].append(float(pe["rot_deg"]))
            per_method[m]["trans_rel"].append(float(pe["trans"]) / ref_depth)
            if c == 0:
                traj[m][t] = est[m]["T"]
        if c == 0:
            have[t] = True
    c2w_gt = geo.invert_se3(geo.rigidify(T_w2c))
    ate = {}
    path = np.ptp(c2w_gt[have, :3, 3], axis=0).max() if have.any() else 0.0
    if have.sum() >= 3 and path > 1e-3 * ref_depth:  # a static camera makes the Sim(3) alignment degenerate
        for m in POSE_METHODS:
            r = checks.ate_rpe(traj[m][have], c2w_gt[have], align="sim3")
            ate[m] = {"ate_rel": r["ate"] / ref_depth, "rpe_trans_rel": r["rpe_trans"] / ref_depth,
                      "rpe_rot_deg": r["rpe_rot_deg"]}
    return {
        "pairs": {k: np.asarray(v) for k, v in pairs.items()},
        "methods": {m: {k: np.asarray(v) for k, v in vals.items()} for m, vals in per_method.items()},
        "trajectory": ate,
        "poses_to_0": traj,  # per method, [T, 4, 4] T_{t->0} (identity where ``have_pose`` is False)
        "have_pose": have,
        "K_fit": K_fit,
        "ref_depth": ref_depth,
    }
