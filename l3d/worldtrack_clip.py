"""Load one clip of a WorldTrack sequence with GT re-based to the clip's first frame.

Mirrors `load_worldtrack_sequence` in `eval_track3d_in_worldtrack.py` (JPEG decode,
pinhole projection, `extrinsics_w2c` re-basing, identity pose when absent), but
for an arbitrary window `[clip_start, clip_start + clip_len)`, so that world =
camera of clip frame 0 (EXPERIMENTS.md section 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval_track3d_in_worldtrack import _decode_jpeg_rgb, _project_points_to_video_frame

# Subsets whose videos come from a fixed camera: identity pose is the correct GT
# even though the npz carries no `extrinsics_w2c`.
FIXED_CAMERA_SUBSETS: frozenset[str] = frozenset({"pstudio_mini"})
STATIC_DISP_RATIO = 0.01


@dataclass
class WorldTrackClip:
    subset: str
    sequence: str
    sequence_path: str
    clip_start: int
    video: np.ndarray  # [T, H, W, 3] uint8, native resolution
    K_gt: np.ndarray  # [3, 3] f64, native pixels
    T_w2c_gt: np.ndarray  # [T, 4, 4] f64, world = clip frame 0
    has_pose_gt: bool  # npz has `extrinsics_w2c`
    fixed_camera: bool  # identity pose is GT (e.g. Panoptic Studio)
    gt_xyz_cam: np.ndarray  # [T, N, 3] f32, track n at frame t in camera t (NaN invalid)
    gt_uv: np.ndarray  # [T, N, 2] f32, normalized by (W - 1, H - 1)
    gt_vis: np.ndarray  # [T, N] bool
    gt_world: np.ndarray  # [T, N, 3] f64, world = camera of clip frame 0
    is_static: np.ndarray  # [N] bool
    median_depth: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.video.shape[0])

    @property
    def N(self) -> int:
        return int(self.gt_xyz_cam.shape[1])

    @property
    def hw_native(self) -> tuple[int, int]:
        return int(self.video.shape[1]), int(self.video.shape[2])

    @property
    def aspect_wh(self) -> float:
        h, w = self.hw_native
        return float(w) / float(max(h, 1))

    @property
    def pose_gt_valid(self) -> bool:
        """True when `T_w2c_gt` is real GT (npz extrinsics, or a fixed camera)."""
        return bool(self.has_pose_gt or self.fixed_camera)

    def gt_point(self, n: Any, t: Any, c: Any) -> np.ndarray:
        """GT point of track `n` at time `t` in camera `c`: `T_w2c[c] @ inv(T_w2c[t]) @ X_t(n)`.

        Vectorized over broadcastable integer arrays. Returns float64 `[..., 3]`,
        NaN where `n < 0` or the GT point is not finite.
        """
        return gt_point(self.gt_world, self.T_w2c_gt, n, t, c)


def gt_point(gt_world: np.ndarray, T_w2c: np.ndarray, n: Any, t: Any, c: Any) -> np.ndarray:
    """`X_c = T_w2c[c] @ inv(T_w2c[t]) @ X_t(n)` using precomputed world points `gt_world[t, n]`."""
    n_arr, t_arr, c_arr = np.broadcast_arrays(
        np.asarray(n, dtype=np.int64), np.asarray(t, dtype=np.int64), np.asarray(c, dtype=np.int64)
    )
    out_shape = n_arr.shape + (3,)
    n_flat = n_arr.reshape(-1)
    t_flat = t_arr.reshape(-1)
    c_flat = c_arr.reshape(-1)
    valid = n_flat >= 0
    world = np.full((n_flat.shape[0], 3), np.nan, dtype=np.float64)
    world[valid] = np.asarray(gt_world, dtype=np.float64)[t_flat[valid], n_flat[valid]]
    rot = np.asarray(T_w2c, dtype=np.float64)[c_flat, :3, :3]
    trans = np.asarray(T_w2c, dtype=np.float64)[c_flat, :3, 3]
    out = np.einsum("mij,mj->mi", rot, world) + trans
    return out.reshape(out_shape)


def _static_mask(gt_world: np.ndarray, median_depth: float) -> np.ndarray:
    # Max displacement of each track from its first finite world position.
    world = np.asarray(gt_world, dtype=np.float64)
    finite = np.isfinite(world).all(axis=-1)  # [T, N]
    num_frames, num_tracks = finite.shape
    has_ref = finite.any(axis=0)
    ref_idx = np.argmax(finite, axis=0)
    ref = world[ref_idx, np.arange(num_tracks)]  # [N, 3]
    dists = np.linalg.norm(world - ref[None], axis=-1)
    dists = np.where(finite, dists, 0.0)
    disp = dists.max(axis=0) if num_frames > 0 else np.zeros((num_tracks,), dtype=np.float64)
    enough = finite.sum(axis=0) >= 2
    if not np.isfinite(median_depth) or median_depth <= 0.0:
        return np.zeros((num_tracks,), dtype=bool)
    return has_ref & enough & (disp < STATIC_DISP_RATIO * float(median_depth))


def load_clip(
    npz_path: str | Path,
    clip_start: int = 0,
    clip_len: int = 48,
    subset: str | None = None,
    fixed_camera: bool | None = None,
) -> WorldTrackClip:
    """Load frames `[clip_start, clip_start + clip_len)` of a WorldTrack `.npz`.

    `subset` defaults to the parent directory name. `fixed_camera` defaults to
    `subset in FIXED_CAMERA_SUBSETS`. If the npz has no `extrinsics_w2c` and the
    camera is not fixed, `T_w2c_gt` is identity (as upstream) but it is NOT GT, so
    `is_static` is all False.
    """
    npz_path = Path(npz_path)
    subset = str(subset) if subset is not None else npz_path.parent.name
    fixed_camera = bool(subset in FIXED_CAMERA_SUBSETS) if fixed_camera is None else bool(fixed_camera)
    clip_start = int(clip_start)
    if clip_start < 0 or int(clip_len) <= 0:
        raise ValueError(f"Invalid clip_start={clip_start} clip_len={clip_len}")

    pack = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
    tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64).reshape(-1)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    has_pose_gt = "extrinsics_w2c" in pack.files
    extrinsics_raw = np.asarray(pack["extrinsics_w2c"], dtype=np.float64) if has_pose_gt else None

    seq_frames = min(int(images_jpeg_bytes.shape[0]), int(tracks_xyz_cam.shape[0]), int(visibility.shape[0]))
    clip_end = min(seq_frames, clip_start + int(clip_len))
    if clip_end <= clip_start:
        raise RuntimeError(f"No usable frames in {npz_path} for clip_start={clip_start} (sequence has {seq_frames})")
    sl = slice(clip_start, clip_end)
    num_frames = clip_end - clip_start

    video = np.stack([_decode_jpeg_rgb(frame_bytes) for frame_bytes in images_jpeg_bytes[sl]], axis=0)
    if video.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 frames in {npz_path}, got {video.dtype}")
    height, width = int(video.shape[1]), int(video.shape[2])

    xyz_cam = tracks_xyz_cam[sl].copy()
    xyz_cam[~np.isfinite(xyz_cam).all(axis=-1)] = np.nan
    vis = visibility[sl].copy()

    uv_px = _project_points_to_video_frame(xyz_cam, intrinsics)
    uv_norm = uv_px.copy()
    uv_norm[..., 0] /= float(max(width - 1, 1))
    uv_norm[..., 1] /= float(max(height - 1, 1))

    fx, fy, cx, cy = intrinsics[:4]
    K_gt = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    if extrinsics_raw is not None:
        extr = extrinsics_raw[sl]
        first_inv = np.linalg.inv(extr[0])
        T_w2c = np.asarray([e @ first_inv for e in extr], dtype=np.float64)
    else:
        T_w2c = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
    T_c2w = np.linalg.inv(T_w2c)
    gt_world = np.einsum("tij,tnj->tni", T_c2w[:, :3, :3], xyz_cam) + T_c2w[:, None, :3, 3]

    # Diagnostic: share of GT-visible samples projecting inside the image (about 1 when
    # `tracks_XYZ` really is in per-frame camera coordinates).
    inside = np.isfinite(uv_norm).all(axis=-1) & (uv_norm >= -0.01).all(axis=-1) & (uv_norm <= 1.01).all(axis=-1)
    vis_inside_frac = float(inside[vis].mean()) if np.any(vis) else float("nan")

    z = xyz_cam[..., 2]
    z_ok = np.isfinite(z) & (z > 0.0)
    median_depth = float(np.median(z[z_ok])) if np.any(z_ok) else float("nan")
    if has_pose_gt or fixed_camera:
        is_static = _static_mask(gt_world, median_depth)
    else:
        is_static = np.zeros((xyz_cam.shape[1],), dtype=bool)

    return WorldTrackClip(
        subset=subset,
        sequence=npz_path.stem,
        sequence_path=str(npz_path),
        clip_start=clip_start,
        video=video,
        K_gt=K_gt,
        T_w2c_gt=T_w2c,
        has_pose_gt=bool(has_pose_gt),
        fixed_camera=fixed_camera,
        gt_xyz_cam=xyz_cam.astype(np.float32),
        gt_uv=uv_norm.astype(np.float32),
        gt_vis=vis,
        gt_world=gt_world,
        is_static=is_static,
        median_depth=median_depth,
        extras={
            "seq_frames": int(seq_frames),
            "intrinsics_fx_fy_cx_cy": intrinsics[:4].copy(),
            "vis_inside_frac": vis_inside_frac,
        },
    )
