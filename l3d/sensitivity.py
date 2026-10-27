"""A2: sensitivity of the five checks to injected errors on a synthetic scene.

A seeded synthetic clip (moving camera, static background, rigid moving objects) is
decoded by a *virtual decoder* ``decode(n, s, t, c)`` that returns the GT point
``X_c(n, t)`` plus one injected error, and a virtual 2D head ``uv_hat(n, s, t)``.
Every check is computed on query sets that mirror EXPERIMENTS.md designs 1-4, once with
the clean decoder and once with the injection; the response is
``(residual_injected - residual_clean) / eps``.

Injected error types (magnitude ``eps``, default 0.05 = 5% of depth or equivalent):

- ``depth_offset``: per-object log-depth offset along the ray (``X * exp(eps a)``).
- ``depth_jitter``: the same, drawn per object and frame.
- ``image_plane``: per-point shift of ``eps * |X|`` perpendicular to the ray.
- ``nonrigid``: smooth random displacement field (random Fourier features, length
  scale 1.5 m), ``eps * z`` RMS, on cross-frame outputs only (``c != t``).
- ``nonrigid_radial``: that field projected on camera t's ray (rescaled by sqrt(3) to
  keep the RMS), the along-ray control for Table 1's "Partly (perp ray)".
- ``gauge``: rigid transform (rotation ``eps / sqrt(2)`` rad, translation
  ``eps / sqrt(2) * median depth``) on cross-frame outputs.
- ``scale``: scale ``exp(eps a)`` about the camera center on cross-frame outputs.

Dependence (which indices the random draw takes):

- ``T``: (unit, t) -- shared by every query of the point at ``t_t``;
- ``S``: (unit, s, t) -- changes with the source frame;
- ``C``: (unit, t, c) -- changes with the camera frame, applied only when ``c != t``.

The unit is the motion group for depth types (offset: group only, constant over t),
the point for image-plane, the whole frame for camera-frame types; the rigid gauge and
per-camera scale ``C`` are drawn per camera frame ``c`` only (one ``E_c`` for all cross
outputs in ``c``). Self-frame types with ``T``/``S`` are applied in camera t and
propagate to cross-frame outputs by the GT pose; with ``C`` they act in camera c (along
or across camera c's ray). Camera-frame types with ``T``/``S`` act in camera t
coordinates before the GT mapping, still only on cross-frame outputs. Draws are
fixed-magnitude (random sign / direction), so responses do not depend on draw size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from l3d import checks
from l3d import geometry as geo

KINDS: tuple[str, ...] = (
    "depth_offset", "depth_jitter", "image_plane", "nonrigid", "nonrigid_radial", "gauge", "scale",
)
DEPS: tuple[str, ...] = ("T", "S", "C")
SELF_KINDS = ("depth_offset", "depth_jitter", "image_plane")
CAMERA_KINDS = ("nonrigid", "nonrigid_radial", "gauge", "scale")

# Check columns of the sensitivity matrix and the raw metrics behind each (max over them).
COLUMNS: dict[str, tuple[str, ...]] = {
    "source_swap": ("source_swap",),
    "source_swap_gt_u2": ("source_swap_gt_u2",),
    "camera_swap_se3": ("camera_swap_se3",),
    "camera_swap_sim3": ("camera_swap_sim3",),
    "ray_center_se3": ("ray_center_se3_spread", "ray_center_se3_center"),
    "ray_center_sim3": ("ray_center_sim3_spread", "ray_center_sim3_center"),
    "ray_center_gt": ("ray_center_gt_spread", "ray_center_gt_center"),
    "reprojection": ("reprojection",),
    "static_rigidity": ("static_rigidity",),
    "static_rigidity_excl_tc": ("static_rigidity_excl_tc",),
}
COLUMN_LABELS: dict[str, str] = {
    "source_swap": "source swap (model u2)",
    "source_swap_gt_u2": "source swap (GT u2)",
    "camera_swap_se3": "camera swap SE(3)",
    "camera_swap_sim3": "camera swap Sim(3)",
    "ray_center_se3": "ray center (SE(3) fit, fitted K)",
    "ray_center_sim3": "ray center (Sim(3) fit, fitted K)",
    "ray_center_gt": "ray center (GT K, R, t)",
    "reprojection": "reprojection (fitted K)",
    "static_rigidity": "static rigidity (incl. t_t=c)",
    "static_rigidity_excl_tc": "static rigidity (excl. t_t=c)",
}
# Table 1 column each matrix column is scored against (None = not in Table 1).
TABLE_COLUMN: dict[str, str | None] = {
    "source_swap": "source_swap",
    "source_swap_gt_u2": "source_swap",
    "camera_swap_se3": "camera_swap",
    "camera_swap_sim3": "camera_swap",
    "ray_center_se3": "ray_center",
    "ray_center_sim3": "ray_center",
    "ray_center_gt": None,
    "reprojection": "reprojection",
    "static_rigidity": "static_rigidity",
    "static_rigidity_excl_tc": "static_rigidity",
}

# Table 1 (tab:vis) of the proposal, verbatim entries; footnotes resolved in predicted_class.
TABLE1: dict[tuple[str, str], dict[str, str]] = {
    ("image_plane", "T"): {"source_swap": "No^a", "camera_swap": "No", "ray_center": "Yes", "reprojection": "Yes", "static_rigidity": "Varying part"},
    ("image_plane", "S"): {"source_swap": "Yes", "camera_swap": "No", "ray_center": "Yes (t_s=t_t)", "reprojection": "Yes (t_s=t_t)", "static_rigidity": "No"},
    ("depth_jitter", "T"): {"source_swap": "No", "camera_swap": "No", "ray_center": "No", "reprojection": "No", "static_rigidity": "Static"},
    ("depth_jitter", "S"): {"source_swap": "Yes", "camera_swap": "No", "ray_center": "No", "reprojection": "No", "static_rigidity": "No"},
    ("depth_offset", "T"): {"source_swap": "No", "camera_swap": "No", "ray_center": "No", "reprojection": "No", "static_rigidity": "Static^b"},
    ("nonrigid", "C"): {"source_swap": "No", "camera_swap": "Yes", "ray_center": "Partly (perp ray)", "reprojection": "No", "static_rigidity": "Static"},
    ("gauge", "C"): {"source_swap": "No", "camera_swap": "No", "ray_center": "No", "reprojection": "No", "static_rigidity": "Static^c"},
    # Caption: "A per-camera-frame scale is invisible to every check under a Sim(3) Umeyama
    # fit (camera swap sees it under SE(3))."
    ("scale", "C"): {"source_swap": "No (Sim3 note)", "camera_swap": "SE3 Yes / Sim3 No", "ray_center": "Sim3 No", "reprojection": "No (Sim3 note)", "static_rigidity": "No (Sim3 note)"},
}
# Not a Table 1 row: the along-ray control implied by "Partly: only the part perpendicular to the ray".
DERIVED: dict[tuple[str, str], dict[str, str]] = {
    ("nonrigid_radial", "C"): {"source_swap": "No", "camera_swap": "Yes", "ray_center": "No (along ray)", "reprojection": "No", "static_rigidity": "Static"},
}


# --------------------------------------------------------------------------------------
# Synthetic scene
# --------------------------------------------------------------------------------------


@dataclass
class Scene:
    K: np.ndarray  # [3, 3]
    hw: tuple[int, int]
    T_w2c: np.ndarray  # [T, 4, 4], world = camera of frame 0
    X_world: np.ndarray  # [T, N, 3]
    group: np.ndarray  # [N], 0 = static background, 1..G objects
    R_obj: np.ndarray  # [G + 1, T, 3, 3] world rotation of each group (identity for static)
    vis: np.ndarray  # [T, N]
    translates: bool
    X_cam: np.ndarray = field(init=False)  # [T, N, 3]
    uv_px: np.ndarray = field(init=False)  # [T, N, 2]
    T_c2w: np.ndarray = field(init=False)
    T_rel: np.ndarray = field(init=False)  # [T, T, 4, 4], camera t -> camera c

    def __post_init__(self) -> None:
        self.T_c2w = geo.invert_se3(self.T_w2c)
        self.X_cam = geo.apply_transform(self.T_w2c[:, None], self.X_world)
        self.uv_px = geo.project(self.K, self.X_cam)
        n_t = self.T_w2c.shape[0]
        self.T_rel = geo.relative_pose(self.T_w2c, np.arange(n_t)[:, None], np.arange(n_t)[None, :])

    @property
    def T(self) -> int:
        return int(self.T_w2c.shape[0])

    @property
    def N(self) -> int:
        return int(self.X_world.shape[1])

    @property
    def is_static(self) -> np.ndarray:
        return self.group == 0

    @property
    def median_depth(self) -> float:
        return float(np.median(self.X_cam[..., 2]))


def _visible_all(K: np.ndarray, hw: tuple[int, int], X_cam: np.ndarray, zmin: float, zmax: float, margin: float = 3.0) -> np.ndarray:
    """``[N]`` mask: in front, within ``[zmin, zmax]`` and inside the image in every frame."""
    uv = geo.project(K, X_cam)
    h, w = hw
    ok = (X_cam[..., 2] > zmin) & (X_cam[..., 2] < zmax)
    ok &= (uv[..., 0] > margin) & (uv[..., 0] < w - 1 - margin)
    ok &= (uv[..., 1] > margin) & (uv[..., 1] < h - 1 - margin)
    return ok.all(0)


def _camera_path(rng: np.random.Generator, T: int, translates: bool) -> np.ndarray:
    """``T_w2c [T, 4, 4]``: sideways/forward translation with an inward yaw, pitch and roll."""
    tau = np.linspace(0.0, 1.0, T)
    side = rng.choice([-1.0, 1.0])
    length = rng.uniform(1.0, 1.6) if translates else 0.0
    direction = np.array([side, rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.6)])
    direction /= np.linalg.norm(direction)
    pos = length * tau[:, None] * direction + np.outer(np.sin(np.pi * tau), [0.0, 0.05 * length, 0.0])
    yaw = -side * np.radians(rng.uniform(8.0, 14.0))
    pitch = np.radians(rng.uniform(-4.0, 4.0))
    roll = np.radians(rng.uniform(-2.0, 2.0))
    rot = geo.rotvec_to_matrix(tau[:, None] * np.array([pitch, yaw, roll]))
    c2w = np.tile(np.eye(4), (T, 1, 1))
    c2w[:, :3, :3] = rot
    c2w[:, :3, 3] = pos
    return geo.invert_se3(c2w)


def _static_candidates(rng: np.random.Generator, n: int) -> np.ndarray:
    """Points on a back wall, the floor and two boxes (world = camera 0, +y down)."""
    bx, bz = rng.uniform(-0.3, 0.3, 2)
    pick = rng.choice(5, size=n, p=[0.35, 0.3, 0.12, 0.12, 0.11])
    u, v = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    pts = np.zeros((n, 3))
    m = pick == 0  # back wall, slightly tilted
    pts[m] = np.c_[-5 + 10 * u[m], -3 + 4.4 * v[m], 7.5 + 0.12 * (-5 + 10 * u[m])]
    m = pick == 1  # floor
    pts[m] = np.c_[-4 + 8 * u[m], np.full(m.sum(), 1.4), 2.2 + 5.3 * v[m]]
    m = pick == 2  # box A front face
    pts[m] = np.c_[-2.6 + bx + 1.2 * u[m], 0.4 + 1.0 * v[m], np.full(m.sum(), 4.0 + bz)]
    m = pick == 3  # box B front face
    pts[m] = np.c_[1.3 + bx + 1.2 * u[m], -0.2 + 1.6 * v[m], np.full(m.sum(), 5.5 + bz)]
    m = pick == 4  # box B side face
    pts[m] = np.c_[np.full(m.sum(), 1.3 + bx), -0.2 + 1.6 * v[m], 5.5 + bz + 1.0 * u[m]]
    return pts


def make_scene(
    seed: int = 0,
    T: int = 24,
    n_static: int = 400,
    n_objects: int = 3,
    n_per_object: int = 150,
    focal: float = 220.0,
    hw: tuple[int, int] = (256, 256),
    camera_translates: bool = True,
    occlusion_prob: float = 0.0,
) -> Scene:
    """Seeded synthetic clip; every point is inside the image at 2-8 m in all frames.

    ``occlusion_prob`` > 0 marks random ``(t, n)`` as not visible (frame 0 stays visible).
    """
    rng = np.random.default_rng(seed)
    h, w = hw
    K = np.array([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    for _ in range(50):
        T_w2c = _camera_path(rng, T, camera_translates)
        cand = _static_candidates(rng, 6 * n_static)
        Xc = geo.apply_transform(T_w2c[:, None], np.broadcast_to(cand, (T,) + cand.shape))
        ok = _visible_all(K, hw, Xc, 1.5, 9.0) & (Xc[0, :, 2] > 2.0) & (Xc[0, :, 2] < 8.0)
        if ok.sum() >= n_static:
            break
    else:
        raise RuntimeError("could not place static points; widen the scene")
    static = cand[rng.choice(np.flatnonzero(ok), n_static, replace=False)]
    X_world = [np.broadcast_to(static, (T, n_static, 3))]
    R_obj = [np.tile(np.eye(3), (T, 1, 1))]

    tau = np.linspace(0.0, 1.0, T)
    for _ in range(n_objects):
        for _ in range(200):
            axes = rng.uniform(0.25, 0.45, 3)
            center0 = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.6, 0.6), rng.uniform(3.0, 5.0)])
            vel = rng.normal(size=3)
            vel *= rng.uniform(0.4, 0.9) / np.linalg.norm(vel)
            axis = rng.normal(size=3)
            axis *= np.radians(rng.uniform(25.0, 60.0)) / np.linalg.norm(axis)
            dirs = rng.normal(size=(n_per_object, 3))
            local = axes * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            R_t = geo.rotvec_to_matrix(tau[:, None] * axis)
            centers = center0 + tau[:, None] * vel
            Xw = centers[:, None] + np.einsum("tij,nj->tni", R_t, local)
            Xc = geo.apply_transform(T_w2c[:, None], Xw)
            if _visible_all(K, hw, Xc, 1.5, 9.0).all():
                break
        else:
            raise RuntimeError("could not place a moving object inside the view")
        X_world.append(Xw)
        R_obj.append(R_t)

    X_world = np.concatenate(X_world, axis=1).astype(np.float64)
    group = np.concatenate([np.zeros(n_static, np.int64)] + [np.full(n_per_object, g + 1) for g in range(n_objects)])
    vis = np.ones((T, X_world.shape[1]), dtype=bool)
    if occlusion_prob > 0:
        vis = rng.random(vis.shape) >= occlusion_prob
        vis[0] = True
    return Scene(K=K, hw=hw, T_w2c=T_w2c, X_world=X_world, group=group, R_obj=np.stack(R_obj), vis=vis,
                 translates=camera_translates)


# --------------------------------------------------------------------------------------
# Virtual decoder
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Injection:
    kind: str = "none"
    dep: str = "T"
    eps: float = 0.05


def _unit(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    v = rng.normal(size=shape + (3,))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _perp_basis(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.broadcast_to(np.array([0.0, 1.0, 0.0]), r.shape)
    e1 = np.cross(r, up)
    bad = np.linalg.norm(e1, axis=-1) < 1e-6
    e1[bad] = np.cross(r[bad], np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(r, e1)


class VirtualDecoder:
    """``decode(n, s, t, c)`` = GT ``X_c(n, t)`` + one injected error (see module docstring)."""

    def __init__(
        self,
        scene: Scene,
        injection: Injection | None = None,
        seed: int = 0,
        uv_follows_xyz: bool = True,
        n_features: int = 8,
        length_scale: float = 1.5,
    ) -> None:
        self.scene = scene
        self.inj = injection or Injection("none")
        self.uv_follows_xyz = uv_follows_xyz
        self.n_features = n_features
        kind, dep, eps = self.inj.kind, self.inj.dep, self.inj.eps
        if kind != "none" and (kind not in KINDS or dep not in DEPS):
            raise ValueError(f"unknown injection {self.inj}")
        rng = np.random.default_rng(seed)
        T, N, G = scene.T, scene.N, int(scene.group.max()) + 1
        shapes = {
            "depth_offset": {"T": (G,), "S": (G, T), "C": (G, T)},
            "depth_jitter": {"T": (G, T), "S": (G, T, T), "C": (G, T, T)},
            "image_plane": {"T": (N, T), "S": (N, T, T), "C": (N, T, T)},
            "nonrigid": {"T": (T,), "S": (T, T), "C": (T, T)},
            "nonrigid_radial": {"T": (T,), "S": (T, T), "C": (T, T)},
            "gauge": {"T": (T,), "S": (T, T), "C": (T,)},
            "scale": {"T": (T,), "S": (T, T), "C": (T,)},
        }
        self.d: dict[str, np.ndarray] = {}
        if kind == "none":
            return
        shape = shapes[kind][dep]
        if kind in ("depth_offset", "depth_jitter", "scale"):
            self.d["sign"] = rng.choice([-1.0, 1.0], size=shape)
        elif kind == "image_plane":
            self.d["phi"] = rng.uniform(0.0, 2.0 * np.pi, size=shape)
        elif kind in ("nonrigid", "nonrigid_radial"):
            kf = n_features
            self.d["omega"] = rng.normal(0.0, 1.0 / length_scale, size=shape + (kf, 3))
            self.d["phase"] = rng.uniform(0.0, 2.0 * np.pi, size=shape + (kf,))
            self.d["amp"] = rng.normal(0.0, 1.0 / np.sqrt(3.0), size=shape + (kf, 3))
        elif kind == "gauge":
            self.d["R"] = geo.rotvec_to_matrix(eps / np.sqrt(2.0) * _unit(rng, shape))
            self.d["t"] = eps / np.sqrt(2.0) * scene.median_depth * _unit(rng, shape)

    # -- draw keys ---------------------------------------------------------------------

    def _key(self, n: np.ndarray, g: np.ndarray, s: np.ndarray, t: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, ...]:
        kind, dep = self.inj.kind, self.inj.dep
        if kind == "depth_offset":
            return {"T": (g,), "S": (g, s), "C": (g, c)}[dep]
        if kind == "depth_jitter":
            return {"T": (g, t), "S": (g, s, t), "C": (g, t, c)}[dep]
        if kind == "image_plane":
            return {"T": (n, t), "S": (n, s, t), "C": (n, t, c)}[dep]
        if kind in ("nonrigid", "nonrigid_radial"):
            return {"T": (t,), "S": (s, t), "C": (t, c)}[dep]
        return {"T": (t,), "S": (s, t), "C": (c,)}[dep]  # gauge, scale

    # -- error models ------------------------------------------------------------------

    def _self_err(self, Z: np.ndarray, key: tuple[np.ndarray, ...]) -> np.ndarray:
        eps = self.inj.eps
        if self.inj.kind in ("depth_offset", "depth_jitter"):
            return Z * np.exp(eps * self.d["sign"][key])[:, None]
        rng_ = np.linalg.norm(Z, axis=-1, keepdims=True)
        e1, e2 = _perp_basis(Z / rng_)
        phi = self.d["phi"][key][:, None]
        return Z + eps * rng_ * (np.cos(phi) * e1 + np.sin(phi) * e2)

    def _cam_err(self, Z: np.ndarray, key: tuple[np.ndarray, ...], origin: np.ndarray) -> np.ndarray:
        """Camera-frame error on points ``Z``; ``origin`` = camera t's center in Z's frame."""
        kind, eps = self.inj.kind, self.inj.eps
        if kind in ("nonrigid", "nonrigid_radial"):
            arg = np.einsum("mkj,mj->mk", self.d["omega"][key], Z) + self.d["phase"][key]
            f = np.sqrt(2.0 / self.n_features) * np.einsum("mk,mkj->mj", np.cos(arg), self.d["amp"][key])
            D = eps * Z[:, 2:3] * f
            if kind == "nonrigid_radial":
                ray = Z - origin
                ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
                D = np.sqrt(3.0) * (D * ray).sum(-1, keepdims=True) * ray
            return Z + D
        if kind == "gauge":
            return np.einsum("mij,mj->mi", self.d["R"][key], Z) + self.d["t"][key]
        return Z * np.exp(eps * self.d["sign"][key])[:, None]  # scale

    def _source_offset(self, n: np.ndarray, g: np.ndarray, s: np.ndarray, t: np.ndarray, du_px: np.ndarray) -> np.ndarray:
        """Physical offset (camera t) of the point seen at pixel ``u_gt(n, s) + du``.

        Locally fronto-parallel surface at frame s, carried rigidly with the point's group.
        """
        sc = self.scene
        z_s = sc.X_cam[s, n, 2]
        dXs = np.stack([du_px[:, 0] / sc.K[0, 0] * z_s, du_px[:, 1] / sc.K[1, 1] * z_s, np.zeros_like(z_s)], -1)
        dXw = np.einsum("mij,mj->mi", sc.T_c2w[s, :3, :3], dXs)
        R_move = sc.R_obj[g, t] @ np.swapaxes(sc.R_obj[g, s], -1, -2)
        dXw = np.einsum("mij,mj->mi", R_move, dXw)
        return np.einsum("mij,mj->mi", sc.T_w2c[t, :3, :3], dXw)

    # -- public ------------------------------------------------------------------------

    def decode(self, n, s, t, c, du_px: np.ndarray | None = None) -> np.ndarray:
        """Decoded point at ``t_t = t`` in camera ``c`` from track ``n``'s pixel at ``t_s = s``.

        Index arguments broadcast; ``du_px`` (``[..., 2]``) offsets the source pixel from
        the GT pixel of ``n`` at ``s``. Returns ``[..., 3]``.
        """
        n, s, t, c = np.broadcast_arrays(*(np.asarray(a, dtype=np.int64) for a in (n, s, t, c)))
        shape = n.shape
        n, s, t, c = (a.reshape(-1) for a in (n, s, t, c))
        sc = self.scene
        g = sc.group[n]
        X = sc.X_cam[t, n].copy()
        if du_px is not None:
            X = X + self._source_offset(n, g, s, t, np.asarray(du_px, dtype=np.float64).reshape(-1, 2))
        kind, dep = self.inj.kind, self.inj.dep
        cross = c != t
        key = self._key(n, g, s, t, c) if kind != "none" else ()
        if kind != "none" and dep in ("T", "S"):
            if kind in SELF_KINDS:
                X = self._self_err(X, key)
            elif cross.any():
                X[cross] = self._cam_err(X[cross], tuple(k[cross] for k in key), np.zeros(3))
        Y = geo.apply_transform(sc.T_rel[t, c], X)
        if kind != "none" and dep == "C" and cross.any():
            kc = tuple(k[cross] for k in key)
            if kind in SELF_KINDS:
                Y[cross] = self._self_err(Y[cross], kc)
            else:
                Y[cross] = self._cam_err(Y[cross], kc, sc.T_rel[t[cross], c[cross], :3, 3])
        return Y.reshape(shape + (3,))

    def uv_hat(self, n, s, t) -> np.ndarray:
        """Virtual 2D head: pixel of track ``n`` (from frame ``s``) at frame ``t``.

        GT pixel, or the projection of the decoded self-frame point when ``uv_follows_xyz``.
        """
        if self.uv_follows_xyz:
            return geo.project(self.scene.K, self.decode(n, s, t, t))
        n, t = np.broadcast_arrays(np.asarray(n), np.asarray(t))
        return self.scene.uv_px[t, n]


# --------------------------------------------------------------------------------------
# Query sets (mirror EXPERIMENTS.md designs 1-4)
# --------------------------------------------------------------------------------------


def build_queries(
    scene: Scene,
    seed: int = 0,
    max_tracks: int = 256,
    n_targets: int = 8,
    n_sources: int = 6,
    n_cams: int = 6,
    grid_offset: int = 4,
    max_static: int = 256,
) -> dict[str, Any]:
    """Query sets for the five checks.

    - ``source_swap`` (designs 1 + 4): per track and each of ``n_targets`` evenly spaced
      ``t3``: ``n_sources`` visible sources (always ``t3``), ``n_cams`` cameras (always
      ``t3`` and 0); one reference source ``t1`` per cell, pairs ``(t1, t2)`` for the other
      sources, every camera.
    - ``grid`` (design 2): for every ``t``, all visible points, ``t_s = t_t = t``, decoded in
      ``t`` and in ``c in {0, t - 4, t + 4}`` (clipped, ``c != t``).
    - ``static`` (design 3): GT-static tracks, ``t_s`` = first visible frame, every ``t_t``,
      ``t_c in {0, T // 2}``.
    - ``self``: every visible ``(n, t)`` with ``t_s = t_t = t_c`` (reprojection, intrinsics fit).
    """
    rng = np.random.default_rng(seed)
    T, N = scene.T, scene.N
    vis = scene.vis
    tracks = np.sort(rng.choice(N, size=min(max_tracks, N), replace=False))
    targets = np.unique(np.linspace(0, T - 1, n_targets).round().astype(np.int64))
    rows = []
    for n in tracks:
        visible = np.flatnonzero(vis[:, n])
        for t3 in targets:
            others = visible[visible != t3]
            k = min(n_sources - int(vis[t3, n]), others.size)
            srcs = rng.choice(others, size=k, replace=False)
            if vis[t3, n]:
                srcs = np.r_[t3, srcs]
            if srcs.size < 2:
                continue
            fixed = np.unique([t3, 0])
            rest = np.setdiff1d(np.arange(T), fixed)
            cams = np.r_[fixed, rng.choice(rest, size=max(n_cams - fixed.size, 0), replace=False)]
            t1 = rng.choice(srcs)
            for t2 in srcs[srcs != t1]:
                for c in cams:
                    rows.append((n, t1, t2, t3, c))
    ss = np.array(rows, dtype=np.int64).reshape(-1, 5)

    grid = []
    for t in range(T):
        n_vis = np.flatnonzero(vis[t])
        cams = sorted({int(np.clip(c, 0, T - 1)) for c in (0, t - grid_offset, t + grid_offset)} - {t})
        for c in cams:
            grid.append((t, c, n_vis))

    static_tracks = np.flatnonzero(scene.is_static)
    static_tracks = np.sort(rng.choice(static_tracks, size=min(max_static, static_tracks.size), replace=False))
    first_vis = np.argmax(vis[:, static_tracks], axis=0)

    sf_t, sf_n = np.nonzero(vis)
    return {
        "source_swap": {"n": ss[:, 0], "t1": ss[:, 1], "t2": ss[:, 2], "t3": ss[:, 3], "c": ss[:, 4]},
        "grid": grid,
        "static": {"n": static_tracks, "s": first_vis, "cams": sorted({0, T // 2})},
        "self": {"n": sf_n, "t": sf_t},
    }


# --------------------------------------------------------------------------------------
# Residuals of all checks for one decoder
# --------------------------------------------------------------------------------------


def compute_residuals(scene: Scene, dec: VirtualDecoder, q: dict[str, Any]) -> dict[str, float]:
    """Mean residual of every check variant (keys = raw metrics of :data:`COLUMNS`)."""
    out: dict[str, float] = {}
    fx = float(scene.K[0, 0])

    # (1) source swap, model u2 from the virtual 2D head and GT u2 control.
    ss = q["source_swap"]
    P_a = dec.decode(ss["n"], ss["t1"], ss["t3"], ss["c"])
    du = dec.uv_hat(ss["n"], ss["t1"], ss["t2"]) - scene.uv_px[ss["t2"], ss["n"]]
    P_b = dec.decode(ss["n"], ss["t2"], ss["t3"], ss["c"], du_px=du)
    P_b_gt = dec.decode(ss["n"], ss["t2"], ss["t3"], ss["c"])
    out["source_swap"] = float(np.mean(checks.source_swap(P_a, P_b)))
    out["source_swap_gt_u2"] = float(np.mean(checks.source_swap(P_a, P_b_gt)))
    out["u2_px_err"] = float(np.mean(np.linalg.norm(du, axis=-1)))

    # (4) reprojection of self-frame outputs with K fitted from the same outputs.
    sf = q["self"]
    P_self = dec.decode(sf["n"], sf["t"], sf["t"], sf["t"])
    uv_self = scene.uv_px[sf["t"], sf["n"]]
    K_fit = geo.fit_intrinsics_from_points(P_self, uv_self, scene.hw)
    rep = checks.reprojection(P_self, K_fit, uv_self)
    out["reprojection_px"] = float(np.mean(rep))
    out["reprojection"] = out["reprojection_px"] / fx
    out["K_fit_rel_err"] = float(abs(K_fit[0, 0] / fx - 1.0))

    # (2) camera swap a = t -> b = c and (3) ray center, per (t, c) grid pair.
    acc: dict[str, list[np.ndarray]] = {}

    def _add(k: str, v: np.ndarray | float) -> None:
        acc.setdefault(k, []).append(np.atleast_1d(np.asarray(v, dtype=np.float64)))

    for t, c, n in q["grid"]:
        Pa = dec.decode(n, t, t, t)
        Pb = dec.decode(n, t, t, c)
        uv = scene.uv_px[t, n]
        for name, with_scale in (("se3", False), ("sim3", True)):
            cs = checks.camera_swap(Pa, Pb, with_scale=with_scale)
            _add(f"camera_swap_{name}", cs["residual"])
            rc = checks.ray_center(uv, Pb, K_fit, cs["R"], cs["t"])
            _add(f"ray_center_{name}_spread", rc["spread"])
            _add(f"ray_center_{name}_center", rc["center_dist"])
        T_rel = scene.T_rel[t, c]
        rc = checks.ray_center(uv, Pb, scene.K, T_rel[:3, :3], T_rel[:3, 3])
        _add("ray_center_gt_spread", rc["spread"])
        _add("ray_center_gt_center", rc["center_dist"])
    for k, v in acc.items():
        out[k] = float(np.mean(np.concatenate(v)))

    # (5) static rigidity, with and without the query t_t = c.
    st = q["static"]
    tt = np.arange(scene.T)
    incl, excl, at_tc = [], [], []
    for c in st["cams"]:
        P = dec.decode(st["n"][:, None], st["s"][:, None], tt[None, :], c)
        m = scene.vis[tt[None, :], st["n"][:, None]]
        r_incl = checks.static_rigidity(P, m)["residual"]
        incl.append(r_incl)
        at_tc.append(r_incl[:, c])
        excl.append(checks.static_rigidity(P, m & (tt[None, :] != c))["residual"])
    out["static_rigidity"] = float(np.nanmean(np.concatenate([r.ravel() for r in incl])))
    out["static_rigidity_excl_tc"] = float(np.nanmean(np.concatenate([r.ravel() for r in excl])))
    # Diagnostic: residual of the single query with t_t = c (not a matrix column).
    out["static_rigidity_at_tc"] = float(np.nanmean(np.concatenate(at_tc)))
    return out


# --------------------------------------------------------------------------------------
# Predictions and classification
# --------------------------------------------------------------------------------------


def classify(x: float, yes: float = 0.1, partly: float = 0.01) -> str:
    """``Yes`` (>= yes), ``Partly`` (>= partly), ``No`` otherwise."""
    if not np.isfinite(x):
        return "n/a"
    return "Yes" if x >= yes else ("Partly" if x >= partly else "No")


def table_entry(kind: str, dep: str, column: str) -> tuple[str | None, str | None]:
    """Raw Table 1 (or derived) entry for a matrix cell, and its source tag."""
    tcol = TABLE_COLUMN[column]
    if tcol is None:
        return None, None
    if (kind, dep) in TABLE1:
        return TABLE1[(kind, dep)][tcol], "table1"
    if (kind, dep) in DERIVED:
        return DERIVED[(kind, dep)][tcol], "derived"
    return None, None


def predicted_class(kind: str, dep: str, column: str, ctx: dict[str, bool]) -> str | None:
    """Resolve the Table 1 entry (footnotes a, b, c and the Sim(3) note) to Yes/Partly/No."""
    entry, _ = table_entry(kind, dep, column)
    if entry is None:
        return None
    if (kind, dep) == ("scale", "C"):
        if column == "camera_swap_se3":
            return "Yes"
        if column == "ray_center_se3":
            return None  # the note only covers Sim(3) fits for ray center
        return "No"
    if entry == "No^a":  # 2D error of u_hat at t2 leaks in through u2
        return "Yes" if (column == "source_swap" and ctx["uv_follows_xyz"]) else "No"
    if entry == "Static^b":
        return "Yes" if ctx["camera_translates"] else "No"
    if entry == "Static^c":
        return "Yes" if column == "static_rigidity" else "No"
    if entry in ("Static", "Varying part", "Yes (t_s=t_t)"):
        # Static: every injection also hits the static background; Varying part: the
        # injected image-plane T error is i.i.d. over t, so all of it varies.
        return "Yes"
    if entry.startswith("Partly"):
        return "Partly"
    if entry.startswith("No"):
        return "No"
    if entry == "Yes":
        return "Yes"
    raise ValueError(f"unresolved Table 1 entry '{entry}'")


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


def scene_stats(scene: Scene, q: dict[str, Any]) -> dict[str, float]:
    """Baseline/depth ratio seen by static rigidity (drives footnote b), depth range."""
    ratios = []
    z_med = float(np.median(scene.X_cam[:, scene.is_static, 2]))
    for c in q["static"]["cams"]:
        cen = scene.T_rel[:, c, :3, 3]  # camera t's center in camera c, over t
        ratios.append(np.linalg.norm(cen - cen.mean(0), axis=-1).mean() / z_med)
    z = scene.X_cam[..., 2]
    path = scene.T_c2w[:, :3, 3]
    return {
        "baseline_over_depth": float(np.mean(ratios)),
        "camera_path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()),
        "camera_rotation_deg": float(geo.rotation_angle_deg(scene.T_c2w[-1, :3, :3])),
        "depth_min": float(z.min()),
        "depth_median": float(np.median(z)),
        "depth_max": float(z.max()),
        "n_points": scene.N,
        "n_static": int(scene.is_static.sum()),
    }


def run_sensitivity(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    eps: float = 0.05,
    kinds: tuple[str, ...] = KINDS,
    deps: tuple[str, ...] = DEPS,
    camera_translates: bool = True,
    uv_follows_xyz: bool = True,
    yes: float = 0.1,
    partly: float = 0.01,
    scene_kwargs: dict[str, Any] | None = None,
    query_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Responses of every check to every (kind, dependence) injection, over seeds."""
    scene_kwargs = dict(scene_kwargs or {})
    query_kwargs = dict(query_kwargs or {})
    ctx = {"uv_follows_xyz": uv_follows_xyz, "camera_translates": camera_translates}
    raw: dict[str, list[dict[str, float]]] = {}
    clean_all: list[dict[str, float]] = []
    stats: list[dict[str, float]] = []
    for seed in seeds:
        scene = make_scene(seed, camera_translates=camera_translates, **scene_kwargs)
        q = build_queries(scene, seed=10_000 + seed, **query_kwargs)
        stats.append(scene_stats(scene, q))
        clean = compute_residuals(scene, VirtualDecoder(scene, None, uv_follows_xyz=uv_follows_xyz), q)
        clean_all.append(clean)
        for ki, kind in enumerate(kinds):
            for di, dep in enumerate(deps):
                dec = VirtualDecoder(scene, Injection(kind, dep, eps), seed=1_000 * seed + 10 * ki + di,
                                     uv_follows_xyz=uv_follows_xyz)
                res = compute_residuals(scene, dec, q)
                resp = {k: (res[k] - clean[k]) / eps for k in res}
                raw.setdefault(f"{kind}/{dep}", []).append({"residual": res, "response": resp})

    cells: dict[str, Any] = {}
    for key, per_seed in raw.items():
        kind, dep = key.split("/")
        metrics = {
            m: {
                "mean": float(np.mean([p["response"][m] for p in per_seed])),
                "std": float(np.std([p["response"][m] for p in per_seed])),
                "per_seed": [p["response"][m] for p in per_seed],
            }
            for m in per_seed[0]["response"]
        }
        cols = {}
        for col, ms in COLUMNS.items():
            vals = np.array([max(p["response"][m] for m in ms) for p in per_seed])
            mean = float(vals.mean())
            pred = predicted_class(kind, dep, col, ctx)
            entry, source = table_entry(kind, dep, col)
            obs = classify(mean, yes, partly)
            cols[col] = {
                "mean": mean, "std": float(vals.std()), "per_seed": vals.tolist(),
                "observed": obs, "predicted": pred, "table_entry": entry, "prediction_source": source,
                "agree": None if pred is None else bool(pred == obs),
            }
        cells[key] = {"kind": kind, "dep": dep, "columns": cols, "metrics": metrics,
                      "residuals_per_seed": [p["residual"] for p in per_seed]}
    clean_summary = {k: {"mean": float(np.mean([c[k] for c in clean_all])),
                         "max_abs": float(np.max(np.abs([c[k] for c in clean_all])))} for k in clean_all[0]}
    stat_summary = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    return {
        "config": {"seeds": list(seeds), "eps": eps, "camera_translates": camera_translates,
                   "uv_follows_xyz": uv_follows_xyz, "thresholds": {"yes": yes, "partly": partly},
                   "scene_kwargs": scene_kwargs, "query_kwargs": query_kwargs},
        "scene_stats": stat_summary,
        "clean_residuals": clean_summary,
        "cells": cells,
    }


_S_STATIC = (
    "Static rigidity decodes every t_t from one fixed t_s (first visible frame), so an S-indexed "
    "error e(n, t_s, t_t) still changes with t_t and looks like T-jitter to it; Table 1's 'No' holds "
    "only for S errors that are constant in t_t at fixed t_s (cf. depth_offset/S, which behaves like a T offset here)."
)
_DILUTED = (
    "Only the single query with t_t = c escapes the per-camera {what}, so the mean over the T={T} target "
    "frames is diluted to ~2(T-1)/T^2 = {dil:.3f} of the displacement; that query alone has response "
    "{at_tc:.2f} (static_rigidity_at_tc). Visible per query, weak in a mean."
)
EXPLANATIONS: dict[tuple[str, str, str], str] = {
    ("image_plane", "S", "static_rigidity"): _S_STATIC,
    ("image_plane", "S", "static_rigidity_excl_tc"): _S_STATIC,
    ("depth_jitter", "S", "static_rigidity"): _S_STATIC,
    ("depth_jitter", "S", "static_rigidity_excl_tc"): _S_STATIC,
    ("depth_offset", "T", "static_rigidity"): (
        "Residual = eps |c_t - mean_t c_t| / z with c_t camera t's center in camera c, so the response equals "
        "the baseline/depth ratio ({ratio:.3f} here): footnote b's mechanism is right, but the response is weak "
        "unless the camera translates by a sizeable fraction of the depth."
    ),
    ("depth_offset", "T", "static_rigidity_excl_tc"): "Same as the column with t_t = c: response = baseline/depth ratio ({ratio:.3f}).",
    ("nonrigid", "C", "ray_center_se3"): (
        "Table 1's 'Partly' means only the part perpendicular to frame t's rays is seen, not a weak response: "
        "~2/3 of a random smooth field is perpendicular, so the response is large (GT-K/R spread {gt_spread:.2f} vs "
        "camera swap {cs:.2f}); see nonrigid_radial/C for the along-ray part."
    ),
    ("nonrigid", "C", "ray_center_sim3"): "As for the SE(3) fit: 'Partly' is about which component is seen, not its size.",
    ("nonrigid_radial", "C", "ray_center_se3"): (
        "The along-ray field leaves every point on its true ray (GT-K/R ray center: response {gt:.3f}), but it biases "
        "the Umeyama fit: R* tilts the rays (spread {spread:.2f}) and t* leaves the rays' meeting point (center "
        "distance {center:.2f}), so the label-free ray center inherits camera swap's sensitivity."
    ),
    ("nonrigid_radial", "C", "ray_center_sim3"): "Same as the SE(3) fit (spread {spread:.2f}, center distance {center:.2f}; GT-K/R: {gt:.3f}).",
    ("gauge", "C", "static_rigidity"): _DILUTED,
    ("scale", "C", "static_rigidity"): (
        "Footnote c's mechanism also applies to a per-camera scale: the t_t = c query is not rescaled, so the "
        "'invisible to every check under Sim(3)' note holds for the fit-based checks and for static rigidity "
        "without t_t = c, not with it. " + _DILUTED
    ),
}


def explain(d: dict[str, Any], result: dict[str, Any]) -> str:
    """One-line explanation of a disagreement (numbers filled in from ``result``)."""
    key = (d["kind"], d["dep"], d["column"])
    text = EXPLANATIONS.get(key)
    if text is None:
        return "No explanation recorded; inspect the per-metric responses in the JSON."
    cell = result["cells"][d["cell"]]
    met = cell["metrics"]
    T = int(result["config"]["scene_kwargs"].get("T", 24))
    what = {"gauge": "rigid transform E_c", "scale": "scale"}.get(d["kind"], "error")
    fit = "sim3" if d["column"].endswith("sim3") else "se3"
    vals = {
        "ratio": result["scene_stats"]["baseline_over_depth"],
        "T": T,
        "dil": 2.0 * (T - 1) / T**2,
        "what": what,
        "at_tc": met.get("static_rigidity_at_tc", {}).get("mean", float("nan")),
        "gt_spread": met["ray_center_gt_spread"]["mean"],
        "gt": cell["columns"]["ray_center_gt"]["mean"],
        "cs": met[f"camera_swap_{fit}"]["mean"],
        "spread": met[f"ray_center_{fit}_spread"]["mean"],
        "center": met[f"ray_center_{fit}_center"]["mean"],
    }
    return text.format(**vals)


def disagreements(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every matrix cell whose observed class differs from its (resolved) prediction."""
    out = []
    for key, cell in result["cells"].items():
        for col, v in cell["columns"].items():
            if v["agree"] is False:
                out.append({"cell": key, "kind": cell["kind"], "dep": cell["dep"], "column": col,
                            "observed": v["observed"], "predicted": v["predicted"], "mean": v["mean"],
                            "std": v["std"], "table_entry": v["table_entry"], "source": v["prediction_source"]})
    return out


# --------------------------------------------------------------------------------------
# Decoded-query table from the virtual decoder (EXPERIMENTS.md section 3 schema)
# --------------------------------------------------------------------------------------


def synthetic_table(
    scene: Scene,
    dec: VirtualDecoder,
    seed: int = 0,
    model_scale: float = 1.0,
    max_tracks: int = 64,
    n_targets: int = 8,
    n_sources: int = 6,
    n_cams: int = 6,
) -> dict[str, Any]:
    """A decoded-query table (designs 0-5) produced by ``dec``, for dry runs and tests.

    ``p_xyz`` is the decoded point times ``model_scale`` (the unknown model scale);
    design 2 uses the tracks' pixels instead of a 32x32 grid, so every row has GT.
    Designs 4 and 5 follow ``dump_queries``: anchor ``t1`` = ``t_t`` or the nearest
    source, design 5 decoded from the virtual 2D head's ``u2`` (unclipped).
    ``p_conf_logit`` is minus the depth-normalized 3D error (a perfect confidence).
    """
    rng = np.random.default_rng(seed)
    T, N = scene.T, scene.N
    vis = scene.vis
    rows: list[tuple[int, int, int, int, int]] = []  # design, n, ts, tt, tc
    for n in np.flatnonzero(vis[0]):  # 0: track
        for t in range(T):
            for c in sorted({t, 0}):
                rows.append((0, n, 0, t, c))
    tracks = np.sort(rng.choice(N, size=min(max_tracks, N), replace=False))
    targets = np.unique(np.linspace(0, T - 1, n_targets).round().astype(np.int64))
    swap_groups: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []  # n, t1, t3, sources, cameras
    seen4: set[tuple[int, int, int]] = set()
    for n in tracks:  # 1: cross, 4: selfmatch
        visible = np.flatnonzero(vis[:, n])
        for t3 in targets:
            others = visible[visible != t3]
            srcs = rng.choice(others, size=min(n_sources - int(vis[t3, n]), others.size), replace=False)
            srcs = np.sort(np.r_[t3, srcs] if vis[t3, n] else srcs)
            fixed = np.unique([t3, 0])
            cams = np.r_[fixed, rng.choice(np.setdiff1d(np.arange(T), fixed), size=n_cams - fixed.size, replace=False)]
            for s in srcs:
                for c in cams:
                    rows.append((1, n, s, t3, c))
            if srcs.size == 0:
                continue
            t1 = int(t3) if vis[t3, n] else int(srcs[np.argmin(np.abs(srcs - t3))])
            for t2 in srcs:
                if (int(n), t1, int(t2)) not in seen4:
                    seen4.add((int(n), t1, int(t2)))
                    rows.append((4, n, t1, int(t2), int(t2)))
            swap_groups.append((int(n), t1, int(t3), srcs, cams))
    for t in range(T):  # 2: grid
        for c in sorted({t} | {int(np.clip(x, 0, T - 1)) for x in (0, t - 4, t + 4)}):
            for n in np.flatnonzero(vis[t]):
                rows.append((2, n, t, t, c))
    for n in np.flatnonzero(scene.is_static):  # 3: static
        s0 = int(np.argmax(vis[:, n]))
        for t in range(T):
            for c in sorted({0, T // 2}):
                rows.append((3, n, s0, t, c))
    r = np.array(rows, dtype=np.int64)
    design, n, ts, tt, tc = r.T
    P = dec.decode(n, ts, tt, tc)
    uv_px = dec.uv_hat(n, ts, tt)
    src_px = scene.uv_px[ts, n]
    parent = np.full(r.shape[0], -1, dtype=np.int64)

    # 5: swap, second pass from the design-4 rows' u2 = uv_hat(n, t1, t2).
    parent_of = {(int(a), int(b), int(c)): i for i, (k, a, b, c, _) in enumerate(rows) if k == 4}
    swap = [(nn, int(t2), t3, int(c), parent_of[(nn, t1, int(t2))])
            for nn, t1, t3, srcs, cams in swap_groups for t2 in srcs if t2 != t1 for c in cams]
    if swap:
        n5, ts5, tt5, tc5, par5 = np.array(swap, dtype=np.int64).T
        u2 = uv_px[par5]
        du = u2 - scene.uv_px[ts5, n5]
        P5 = dec.decode(n5, ts5, tt5, tc5, du_px=du)
        uv5 = geo.project(scene.K, dec.decode(n5, ts5, tt5, tt5, du_px=du)) if dec.uv_follows_xyz else scene.uv_px[tt5, n5]
        design, n, ts, tt, tc = (np.r_[a, b] for a, b in zip((design, n, ts, tt, tc), (np.full_like(n5, 5), n5, ts5, tt5, tc5)))
        P, uv_px, src_px, parent = np.r_[P, P5], np.r_[uv_px, uv5], np.r_[src_px, u2], np.r_[parent, par5]

    G = geo.apply_transform(scene.T_rel[tt, tc], scene.X_cam[tt, n])
    uv = geo.uv_px_to_norm(uv_px, scene.hw)
    src = geo.uv_px_to_norm(src_px, scene.hw)
    err = np.linalg.norm(P - G, axis=-1) / G[:, 2]
    return {
        "subset": "synthetic",
        "sequence": f"seed{seed}",
        "clip_start": 0,
        "T": T,
        "hw_native": np.array(scene.hw, dtype=np.int64),
        "K_gt": scene.K.astype(np.float64),
        "T_w2c_gt": scene.T_w2c.astype(np.float64),
        "has_pose_gt": True,
        "gt_xyz_cam": scene.X_cam.astype(np.float32),
        "gt_uv": geo.uv_px_to_norm(scene.uv_px, scene.hw).astype(np.float32),
        "gt_vis": vis.copy(),
        "is_static": scene.is_static.copy(),
        "design": design.astype(np.int8),
        "q_track": n.astype(np.int32),
        "q_parent": parent.astype(np.int32),
        "q_u": src[:, 0].astype(np.float32),
        "q_v": src[:, 1].astype(np.float32),
        "q_ts": ts.astype(np.int16),
        "q_tt": tt.astype(np.int16),
        "q_tc": tc.astype(np.int16),
        "p_xyz": (P * model_scale).astype(np.float32),
        "p_uv": uv.astype(np.float32),
        "p_vis_logit": np.full(n.shape[0], 5.0, dtype=np.float32),
        "p_conf_logit": (-err).astype(np.float32),
        "g_xyz": G.astype(np.float32),
        "g_uv": geo.uv_px_to_norm(scene.uv_px[tt, n], scene.hw).astype(np.float32),
        "g_vis": vis[tt, n].copy(),
        "model_ckpt": "virtual",
        "model_config": f"{dec.inj.kind}/{dec.inj.dep}/eps={dec.inj.eps}",
    }
