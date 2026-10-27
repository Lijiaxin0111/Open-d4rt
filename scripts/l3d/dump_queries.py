#!/usr/bin/env python3
"""Dump decoded-query tables for WorldTrack clips (docs/l3d/EXPERIMENTS.md sections 3-4).

For each sequence and clip start: load the clip (world = clip frame 0), encode
once, build the query designs of section 4 with seeded sampling, decode every
query against the cached memory (design 5 in a second pass, from the design-4
predictions), attach GT, and write one `.npz` per clip:
`<out-root>/<run-name>/<subset>/<sequence>__c<clip_start>.npz`, plus
`<out-root>/<run-name>/manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_track3d_in_worldtrack import _compute_average_pts_within_thresh, _compute_scale_factor_global  # noqa: E402
from l3d.engine import QueryEngine  # noqa: E402
from l3d.worldtrack_clip import WorldTrackClip, load_clip  # noqa: E402
from src.core import build_logger, seed_everything  # noqa: E402

DESIGN_NAMES: dict[int, str] = {0: "track", 1: "cross", 2: "grid", 3: "static", 4: "selfmatch", 5: "swap"}

# Section 4 constants.
CROSS_MAX_TRACKS = 256
CROSS_NUM_TARGETS = 8
CROSS_NUM_SOURCES = 6
CROSS_NUM_CAMERAS = 6
GRID_SIZE = 32
GRID_CAM_OFFSET = 4
STATIC_MAX_TRACKS = 256
MIN_DEPTH = 1e-8

QUERY_FIELDS: tuple[str, ...] = ("design", "q_track", "q_parent", "q_u", "q_v", "q_ts", "q_tt", "q_tc")
REQUIRED_KEYS: dict[str, Any] = {
    "subset": np.str_,
    "sequence": np.str_,
    "clip_start": np.int64,
    "T": np.int64,
    "hw_native": np.int64,
    "K_gt": np.float64,
    "T_w2c_gt": np.float64,
    "has_pose_gt": np.bool_,
    "gt_xyz_cam": np.float32,
    "gt_uv": np.float32,
    "gt_vis": np.bool_,
    "is_static": np.bool_,
    "design": np.int8,
    "q_track": np.int32,
    "q_parent": np.int32,
    "q_u": np.float32,
    "q_v": np.float32,
    "q_ts": np.int16,
    "q_tt": np.int16,
    "q_tc": np.int16,
    "p_xyz": np.float32,
    "p_uv": np.float32,
    "p_vis_logit": np.float32,
    "p_conf_logit": np.float32,
    "g_xyz": np.float32,
    "g_uv": np.float32,
    "g_vis": np.bool_,
    "model_ckpt": np.str_,
    "model_config": np.str_,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump decoded-query tables (L3D Part A) for WorldTrack clips.")
    parser.add_argument(
        "--model-config",
        default="checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml",
        help="Model config yaml.",
    )
    parser.add_argument("--ckpt-path", default="", help="Checkpoint path (ignored with --fake-model-tiny).")
    parser.add_argument("--data-root", default="data/worldtrack_release", help="WorldTrack root directory.")
    parser.add_argument(
        "--subsets",
        default="adt_mini,po_mini,pstudio_mini,ds_mini",
        help="Comma-separated WorldTrack subsets.",
    )
    parser.add_argument("--limit-seqs", type=int, default=0, help="Optional cap per subset. <=0 disables.")
    parser.add_argument("--clip-start", default="0", help="Clip start frame(s), comma-separated, e.g. '0' or '0,48'.")
    parser.add_argument("--clip-len", type=int, default=48, help="Clip length T (<= model clip_frames).")
    parser.add_argument(
        "--designs", default="0,1,2,3,4,5", help="Comma-separated query design ids (section 4; 5 needs 1 and 4)."
    )
    parser.add_argument("--out-root", default="output/l3d/dumps", help="Results root.")
    parser.add_argument("--run-name", default="", help="Run name; default derives from the model config.")
    parser.add_argument("--save-memory", action="store_true", help="Also save encoder memory (.pt, fp16) per clip.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=0, help="Seed for query sampling.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Decoder query chunk size.")
    parser.add_argument("--amp", default="none", choices=("none", "bf16", "fp16"), help="CUDA autocast (default off).")
    parser.add_argument("--ckpt-mmap", action="store_true", help="Memory-map the checkpoint (lower peak RAM).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute clips whose table already exists.")
    parser.add_argument(
        "--fake-model-tiny",
        action="store_true",
        help="Use a tiny random-init model (no checkpoint) for CPU smoke tests.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- query designs
def _clip_rng(seed: int, clip: WorldTrackClip, design: int) -> np.random.Generator:
    key = zlib.crc32(f"{clip.subset}/{clip.sequence}".encode("utf-8"))
    return np.random.default_rng([int(seed), int(key), int(clip.clip_start), int(design)])


def _dedupe(values: list[int]) -> list[int]:
    out: list[int] = []
    for item in values:
        if int(item) not in out:
            out.append(int(item))
    return out


def _stratified_pick(candidates: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Pick `k` items from sorted `candidates`, one uniformly at random from each of `k` contiguous bins."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if k <= 0 or candidates.size == 0:
        return []
    if candidates.size <= k:
        return [int(x) for x in candidates]
    return [int(rng.choice(part)) for part in np.array_split(candidates, k)]


def _make_queries(
    design: int,
    track: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ts: np.ndarray,
    tt: np.ndarray,
    tc: np.ndarray,
    parent: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    track = np.asarray(track, dtype=np.int64).reshape(-1)
    parent = np.full(track.shape, -1, dtype=np.int64) if parent is None else np.asarray(parent, dtype=np.int64).reshape(-1)
    return {
        "design": np.full(track.shape, int(design), dtype=np.int8),
        "q_track": track.astype(np.int32),
        "q_parent": parent.astype(np.int32),
        "q_u": np.clip(np.asarray(u, dtype=np.float32).reshape(-1), 0.0, 1.0),
        "q_v": np.clip(np.asarray(v, dtype=np.float32).reshape(-1), 0.0, 1.0),
        "q_ts": np.asarray(ts, dtype=np.int64).reshape(-1).astype(np.int16),
        "q_tt": np.asarray(tt, dtype=np.int64).reshape(-1).astype(np.int16),
        "q_tc": np.asarray(tc, dtype=np.int64).reshape(-1).astype(np.int16),
    }


def _empty_queries(design: int) -> dict[str, np.ndarray]:
    empty_i = np.zeros((0,), dtype=np.int64)
    empty_f = np.zeros((0,), dtype=np.float32)
    return _make_queries(design, empty_i, empty_f, empty_f, empty_i, empty_i, empty_i)


def valid_source_mask(clip: WorldTrackClip) -> np.ndarray:
    """[T, N] frames usable as a query source: GT-visible, finite depth, GT pixel inside the image."""
    uv = clip.gt_uv
    z = clip.gt_xyz_cam[..., 2]
    inside = np.isfinite(uv).all(axis=-1) & (uv >= 0.0).all(axis=-1) & (uv <= 1.0).all(axis=-1)
    return clip.gt_vis & inside & np.isfinite(z) & (np.abs(z) > MIN_DEPTH)


def build_track_design(clip: WorldTrackClip) -> dict[str, np.ndarray]:
    """Design 0: all frame-0-visible tracks, t_s = 0, every t_t, t_c in {t_t, 0} (one row when t_t = 0).

    Track filter and pixels follow upstream eval: visible at frame 0, finite pixel,
    finite depth with |z| > 1e-8; pixels clipped to [0, 1].
    """
    uv0 = clip.gt_uv[0]
    z0 = clip.gt_xyz_cam[0, :, 2]
    keep = clip.gt_vis[0] & np.isfinite(uv0).all(axis=-1) & np.isfinite(z0) & (np.abs(z0) > MIN_DEPTH)
    tracks = np.flatnonzero(keep)
    pattern_tt: list[int] = []
    pattern_tc: list[int] = []
    for t in range(clip.T):
        for c in _dedupe([t, 0]):
            pattern_tt.append(t)
            pattern_tc.append(c)
    reps = len(pattern_tt)
    track = np.repeat(tracks, reps)
    uv = np.repeat(uv0[tracks], reps, axis=0)
    tt = np.tile(np.asarray(pattern_tt, dtype=np.int64), tracks.size)
    tc = np.tile(np.asarray(pattern_tc, dtype=np.int64), tracks.size)
    return _make_queries(0, track, uv[:, 0], uv[:, 1], np.zeros_like(tt), tt, tc)


def build_cross_groups(clip: WorldTrackClip, rng: np.random.Generator) -> list[tuple[int, int, list[int], list[int]]]:
    """Design-1 structure: list of `(n, t_t, sources, cameras)` groups.

    Tracks: up to 256, sampled uniformly from tracks with >= 6 valid source frames,
    topped up from tracks with 2-5 if needed. Targets: 8 evenly spaced frames.
    Sources: `t_t` if valid, plus stratified-random valid frames (6 total).
    Cameras: `t_t` and 0, plus stratified-random frames (6 total).
    """
    valid = valid_source_mask(clip)
    num_valid = valid.sum(axis=0)
    tier_a = np.flatnonzero(num_valid >= CROSS_NUM_SOURCES)
    tier_b = np.flatnonzero((num_valid >= 2) & (num_valid < CROSS_NUM_SOURCES))
    chosen = list(rng.permutation(tier_a)[:CROSS_MAX_TRACKS])
    if len(chosen) < CROSS_MAX_TRACKS:
        chosen += list(rng.permutation(tier_b)[: CROSS_MAX_TRACKS - len(chosen)])
    tracks = np.sort(np.asarray(chosen, dtype=np.int64))

    num_frames = clip.T
    targets = np.unique(np.round(np.linspace(0, num_frames - 1, min(CROSS_NUM_TARGETS, num_frames))).astype(np.int64))
    all_frames = np.arange(num_frames, dtype=np.int64)
    groups: list[tuple[int, int, list[int], list[int]]] = []
    for n in tracks.tolist():
        vis_frames = np.flatnonzero(valid[:, n])
        for t_t in targets.tolist():
            must_src = [t_t] if valid[t_t, n] else []
            others = vis_frames[vis_frames != t_t]
            sources = sorted(must_src + _stratified_pick(others, CROSS_NUM_SOURCES - len(must_src), rng))
            must_cam = _dedupe([t_t, 0])
            others_cam = all_frames[~np.isin(all_frames, must_cam)]
            cameras = sorted(must_cam + _stratified_pick(others_cam, CROSS_NUM_CAMERAS - len(must_cam), rng))
            groups.append((int(n), int(t_t), sources, cameras))
    return groups


def build_cross_design(clip: WorldTrackClip, groups: list[tuple[int, int, list[int], list[int]]]) -> dict[str, np.ndarray]:
    """Design 1: full crossed grid sources x cameras for every `(n, t_t)` group, GT source pixel."""
    rows: list[tuple[int, int, int, int]] = []
    for n, t_t, sources, cameras in groups:
        for s in sources:
            for c in cameras:
                rows.append((n, s, t_t, c))
    if not rows:
        return _empty_queries(1)
    arr = np.asarray(rows, dtype=np.int64)
    uv = clip.gt_uv[arr[:, 1], arr[:, 0]]
    return _make_queries(1, arr[:, 0], uv[:, 0], uv[:, 1], arr[:, 1], arr[:, 2], arr[:, 3])


def selfmatch_anchor(t_t: int, sources: list[int]) -> int:
    """Design-4 anchor `t1` of a design-1 group: `t_t` if it is a source, else the nearest source (earlier on ties)."""
    if t_t in sources:
        return int(t_t)
    src = np.asarray(sources, dtype=np.int64)
    return int(src[np.argmin(np.abs(src - int(t_t)))])


def build_selfmatch_design(clip: WorldTrackClip, groups: list[tuple[int, int, list[int], list[int]]]) -> dict[str, np.ndarray]:
    """Design 4: for each design-1 group, `t_s = t1` (GT pixel), `t_t = t_c = t2` for every group source `t2`.

    `p_uv` of these rows is the model's own correspondence `u2 = uv_hat(u1, t1, t2, t2)`.
    Duplicate `(n, t1, t2)` rows across groups are dropped.
    """
    seen: set[tuple[int, int, int]] = set()
    rows: list[tuple[int, int, int]] = []
    for n, t_t, sources, _ in groups:
        if not sources:
            continue
        t1 = selfmatch_anchor(t_t, sources)
        for t2 in sources:
            key = (n, t1, int(t2))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    if not rows:
        return _empty_queries(4)
    arr = np.asarray(rows, dtype=np.int64)
    uv = clip.gt_uv[arr[:, 1], arr[:, 0]]
    return _make_queries(4, arr[:, 0], uv[:, 0], uv[:, 1], arr[:, 1], arr[:, 2], arr[:, 2])


def grid_uv(size: int = GRID_SIZE) -> np.ndarray:
    """[size*size, 2] normalized pixel-center grid, row-major (v outer, u inner)."""
    coords = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    uu, vv = np.meshgrid(coords, coords, indexing="xy")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)


def build_grid_design(clip: WorldTrackClip) -> dict[str, np.ndarray]:
    """Design 2: 32x32 grid of frame t (t_s = t_t = t), t_c in {t, 0, t-4, t+4} clipped to [0, T), deduped."""
    grid = grid_uv(GRID_SIZE)
    num_frames = clip.T
    ts_list: list[np.ndarray] = []
    tc_list: list[np.ndarray] = []
    for t in range(num_frames):
        cams = _dedupe([t, 0, max(0, t - GRID_CAM_OFFSET), min(num_frames - 1, t + GRID_CAM_OFFSET)])
        for c in cams:
            ts_list.append(np.full((grid.shape[0],), t, dtype=np.int64))
            tc_list.append(np.full((grid.shape[0],), c, dtype=np.int64))
    ts = np.concatenate(ts_list)
    tc = np.concatenate(tc_list)
    reps = ts.shape[0] // grid.shape[0]
    uv = np.tile(grid, (reps, 1))
    return _make_queries(2, np.full(ts.shape, -1, dtype=np.int64), uv[:, 0], uv[:, 1], ts, ts, tc)


def build_static_design(clip: WorldTrackClip, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Design 3: GT-static tracks (up to 256), t_s = first valid frame, every t_t, t_c in {0, T // 2}."""
    valid = valid_source_mask(clip)
    candidates = np.flatnonzero(clip.is_static & valid.any(axis=0))
    if candidates.size > STATIC_MAX_TRACKS:
        candidates = np.sort(rng.choice(candidates, size=STATIC_MAX_TRACKS, replace=False))
    if candidates.size == 0:
        return _empty_queries(3)
    num_frames = clip.T
    first = np.argmax(valid[:, candidates], axis=0)
    cams = _dedupe([0, num_frames // 2])
    pattern_tt = np.repeat(np.arange(num_frames, dtype=np.int64), len(cams))
    pattern_tc = np.tile(np.asarray(cams, dtype=np.int64), num_frames)
    reps = pattern_tt.shape[0]
    track = np.repeat(candidates, reps)
    ts = np.repeat(first, reps)
    uv = clip.gt_uv[ts, track]
    tt = np.tile(pattern_tt, candidates.size)
    tc = np.tile(pattern_tc, candidates.size)
    return _make_queries(3, track, uv[:, 0], uv[:, 1], ts, tt, tc)


def check_designs(designs: list[int]) -> None:
    for design in designs:
        if design not in DESIGN_NAMES:
            raise ValueError(f"Unknown design id {design}; expected one of {sorted(DESIGN_NAMES)}")
    if 5 in designs and not {1, 4} <= set(designs):
        raise ValueError("Design 5 needs designs 1 and 4 in the same table (its parents and swap partners).")


def build_queries(
    clip: WorldTrackClip,
    designs: list[int],
    seed: int,
    groups: list[tuple[int, int, list[int], list[int]]] | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate the requested first-pass designs (0-4, in id order) into one query table."""
    parts: list[dict[str, np.ndarray]] = []
    if groups is None and (1 in designs or 4 in designs):
        groups = build_cross_groups(clip, _clip_rng(seed, clip, 1))
    for design in sorted(set(designs)):
        if design == 0:
            parts.append(build_track_design(clip))
        elif design == 1:
            parts.append(build_cross_design(clip, groups or []))
        elif design == 2:
            parts.append(build_grid_design(clip))
        elif design == 3:
            parts.append(build_static_design(clip, _clip_rng(seed, clip, 3)))
        elif design == 4:
            parts.append(build_selfmatch_design(clip, groups or []))
        elif design == 5:
            raise ValueError("Design 5 is decoded in a second pass; use decode_designs().")
        else:
            raise ValueError(f"Unknown design id {design}; expected one of {sorted(DESIGN_NAMES)}")
    if not parts:
        return _empty_queries(0)
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in QUERY_FIELDS}


def build_swap_design(
    stage1: dict[str, np.ndarray],
    p_uv: np.ndarray,
    groups: list[tuple[int, int, list[int], list[int]]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Design 5: `P(u2_hat, t2, t_t, c)` with the model's own `u2_hat` from the design-4 rows of `stage1`.

    For each design-1 group `(n, t_t, sources, cameras)` with anchor `t1`, every
    source `t2 != t1` and every camera `c`. `q_parent` is the row of design-4 query
    `(n, t1, t2)` in `stage1`; `q_u, q_v` are its `p_uv` clipped to [0, 1]. Rows
    whose parent `p_uv` is non-finite are dropped. Returns the queries and counts
    (`dropped_nonfinite`, `clipped`: parent `p_uv` outside [0, 1]).
    """
    d4 = np.flatnonzero(stage1["design"] == 4)
    parent_of = {
        (int(n), int(t1), int(t2)): int(i)
        for i, n, t1, t2 in zip(d4, stage1["q_track"][d4], stage1["q_ts"][d4], stage1["q_tt"][d4])
    }
    rows: list[tuple[int, int, int, int, int]] = []
    for n, t_t, sources, cameras in groups:
        if not sources:
            continue
        t1 = selfmatch_anchor(t_t, sources)
        for t2 in sources:
            if t2 == t1:
                continue
            parent = parent_of[(n, t1, int(t2))]
            for c in cameras:
                rows.append((n, int(t2), t_t, int(c), parent))
    if not rows:
        return _empty_queries(5), {"dropped_nonfinite": 0, "clipped": 0}
    arr = np.asarray(rows, dtype=np.int64)
    uv = np.asarray(p_uv, dtype=np.float32)[arr[:, 4]]
    finite = np.isfinite(uv).all(axis=-1)
    clipped = finite & ((uv < 0.0) | (uv > 1.0)).any(axis=-1)
    arr, uv = arr[finite], uv[finite]
    queries = _make_queries(5, arr[:, 0], uv[:, 0], uv[:, 1], arr[:, 1], arr[:, 2], arr[:, 3], parent=arr[:, 4])
    return queries, {"dropped_nonfinite": int(np.count_nonzero(~finite)), "clipped": int(np.count_nonzero(clipped))}


def decode_designs(
    engine: QueryEngine, clip: WorldTrackClip, designs: list[int], seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build and decode `designs` against the clip already encoded in `engine`.

    Pass 1 decodes designs 0-4; pass 2 builds design 5 from the pass-1 design-4
    `uv_2d` and decodes it. Returns `(queries, pred, info)` with rows in design order.
    """
    check_designs(designs)
    groups = build_cross_groups(clip, _clip_rng(seed, clip, 1)) if {1, 4} & set(designs) else None
    queries = build_queries(clip, [d for d in designs if d != 5], seed, groups=groups)
    pred = engine.decode(queries["q_u"], queries["q_v"], queries["q_ts"], queries["q_tt"], queries["q_tc"])
    info: dict[str, Any] = {"decode_s": float(engine.last_decode_seconds)}
    if 5 in designs:
        swap, counts = build_swap_design(queries, pred["uv_2d"], groups or [])
        if swap["design"].shape[0]:
            pred2 = engine.decode(swap["q_u"], swap["q_v"], swap["q_ts"], swap["q_tt"], swap["q_tc"])
            decode2_s = float(engine.last_decode_seconds)
        else:
            pred2, decode2_s = {key: value[:0] for key, value in pred.items()}, 0.0
        info.update({"decode2_s": decode2_s, **{f"swap_{k}": v for k, v in counts.items()}})
        info["decode_s"] += info["decode2_s"]
        queries = {key: np.concatenate([queries[key], swap[key]], axis=0) for key in QUERY_FIELDS}
        pred = {key: np.concatenate([pred[key], pred2[key]], axis=0) for key in pred}
    return queries, pred, info


# ---------------------------------------------------------------------- GT + table
def fill_gt(clip: WorldTrackClip, queries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """GT columns: g_xyz = T_w2c[tc] inv(T_w2c[tt]) X_tt(n), g_uv / g_vis at tt; NaN / False when q_track == -1.

    Without pose GT on a moving camera (identity pose is not GT), cross-frame
    (`q_tc != q_tt`) `g_xyz` rows are NaN.
    """
    track = queries["q_track"].astype(np.int64)
    tt = queries["q_tt"].astype(np.int64)
    tc = queries["q_tc"].astype(np.int64)
    has_track = track >= 0
    g_xyz = clip.gt_point(track, tt, tc).astype(np.float32)
    if not clip.pose_gt_valid:
        g_xyz[tc != tt] = np.nan
    g_uv = np.full((track.shape[0], 2), np.nan, dtype=np.float32)
    g_vis = np.zeros((track.shape[0],), dtype=bool)
    g_uv[has_track] = clip.gt_uv[tt[has_track], track[has_track]]
    g_vis[has_track] = clip.gt_vis[tt[has_track], track[has_track]]
    return {"g_xyz": g_xyz, "g_uv": g_uv, "g_vis": g_vis}


def build_table(
    clip: WorldTrackClip,
    queries: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    model_ckpt: str,
    model_config: str,
    memory_path: str | None,
) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {
        "subset": np.asarray(clip.subset, dtype=np.str_),
        "sequence": np.asarray(clip.sequence, dtype=np.str_),
        "clip_start": np.asarray(clip.clip_start, dtype=np.int64),
        "T": np.asarray(clip.T, dtype=np.int64),
        "hw_native": np.asarray(clip.hw_native, dtype=np.int64),
        "K_gt": np.asarray(clip.K_gt, dtype=np.float64),
        "T_w2c_gt": np.asarray(clip.T_w2c_gt, dtype=np.float64),
        "has_pose_gt": np.asarray(clip.has_pose_gt, dtype=np.bool_),
        "gt_xyz_cam": np.asarray(clip.gt_xyz_cam, dtype=np.float32),
        "gt_uv": np.asarray(clip.gt_uv, dtype=np.float32),
        "gt_vis": np.asarray(clip.gt_vis, dtype=np.bool_),
        "is_static": np.asarray(clip.is_static, dtype=np.bool_),
    }
    table.update({key: queries[key] for key in QUERY_FIELDS})
    table.update(
        {
            "p_xyz": np.asarray(pred["xyz_3d"], dtype=np.float32).reshape(-1, 3),
            "p_uv": np.asarray(pred["uv_2d"], dtype=np.float32).reshape(-1, 2),
            "p_vis_logit": np.asarray(pred["visibility"], dtype=np.float32).reshape(-1),
            "p_conf_logit": np.asarray(pred["confidence"], dtype=np.float32).reshape(-1),
        }
    )
    table.update(gt)
    table["model_ckpt"] = np.asarray(str(model_ckpt), dtype=np.str_)
    table["model_config"] = np.asarray(str(model_config), dtype=np.str_)
    if memory_path:
        table["memory_path"] = np.asarray(str(memory_path), dtype=np.str_)
    return table


def validate_table(table: dict[str, np.ndarray]) -> None:
    """Assert the section-3 schema (keys, dtypes, shapes)."""
    for key, dtype in REQUIRED_KEYS.items():
        if key not in table:
            raise KeyError(f"Missing required key {key}")
        if not np.issubdtype(table[key].dtype, dtype):
            raise TypeError(f"{key}: expected {dtype}, got {table[key].dtype}")
    num_frames = int(table["T"])
    num_tracks = int(table["gt_xyz_cam"].shape[1])
    num_queries = int(table["design"].shape[0])
    expect = {
        "hw_native": (2,),
        "K_gt": (3, 3),
        "T_w2c_gt": (num_frames, 4, 4),
        "gt_xyz_cam": (num_frames, num_tracks, 3),
        "gt_uv": (num_frames, num_tracks, 2),
        "gt_vis": (num_frames, num_tracks),
        "is_static": (num_tracks,),
        "p_xyz": (num_queries, 3),
        "p_uv": (num_queries, 2),
        "g_xyz": (num_queries, 3),
        "g_uv": (num_queries, 2),
    }
    for key in ("q_track", "q_parent", "q_u", "q_v", "q_ts", "q_tt", "q_tc", "p_vis_logit", "p_conf_logit", "g_vis"):
        expect[key] = (num_queries,)
    for key in ("subset", "sequence", "clip_start", "T", "has_pose_gt", "model_ckpt", "model_config"):
        expect[key] = ()
    for key, shape in expect.items():
        if tuple(table[key].shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {tuple(table[key].shape)}")


def _write_npz(path: Path, table: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **table)
    os.replace(tmp, path)


def clip_sanity(table: dict[str, np.ndarray]) -> dict[str, float]:
    """Scale `s` (section 2) and a design-0 APD/EPE (global median scale, ref frame 0) vs upstream protocol."""
    gt = table["g_xyz"].astype(np.float64)
    pred = table["p_xyz"].astype(np.float64)
    self_rows = (table["q_tc"] == table["q_tt"]) & np.isfinite(gt).all(axis=-1) & (table["design"] != 5)
    out = {"scale_selfframe": float(_compute_scale_factor_global(gt[self_rows], pred[self_rows])) if np.any(self_rows) else float("nan")}
    ref_rows = (table["design"] == 0) & (table["q_tc"] == 0)
    if np.any(ref_rows):
        apd, _, _, _, epe = _compute_average_pts_within_thresh(gt[ref_rows], pred[ref_rows], scaling="global", compute_epe=True)
        out.update({"d0_apd_global": float(apd), "d0_epe_global": float(epe)})
    return out


# ---------------------------------------------------------------------- main
def _git_commit() -> dict[str, Any]:
    env_commit = os.environ.get("L3D_GIT_COMMIT", "")
    if env_commit:
        return {"commit": env_commit, "dirty": os.environ.get("L3D_GIT_DIRTY", "unknown")}
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    designs = _parse_int_list(args.designs)
    check_designs(designs)
    clip_starts = _parse_int_list(args.clip_start)
    run_name = args.run_name or (
        f"fake_tiny_T{int(args.clip_len)}" if args.fake_model_tiny else f"{Path(args.model_config).parent.name}_T{int(args.clip_len)}"
    )
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("dump_queries", run_dir)
    seed_everything(int(args.seed), deterministic=True)

    if args.fake_model_tiny:
        engine = QueryEngine.tiny_random(
            args.model_config, clip_frames=int(args.clip_len), device=args.device, chunk_size=int(args.chunk_size), seed=int(args.seed)
        )
        model_ckpt = "fake-model-tiny(random-init)"
    else:
        if not args.ckpt_path:
            raise ValueError("--ckpt-path is required unless --fake-model-tiny is set.")
        engine = QueryEngine.from_checkpoint(
            args.model_config,
            args.ckpt_path,
            device=args.device,
            amp=None if args.amp == "none" else args.amp,
            chunk_size=int(args.chunk_size),
            mmap=bool(args.ckpt_mmap),
            logger=logger,
        )
        model_ckpt = str(args.ckpt_path)
    if int(args.clip_len) > engine.clip_frames:
        raise ValueError(f"--clip-len {args.clip_len} exceeds model clip_frames={engine.clip_frames}")
    logger.info("Device=%s amp=%s clip_frames=%d image_hw=%s", engine.device, engine.amp, engine.clip_frames, engine.image_hw)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"WorldTrack root not found: {data_root}")
    subsets = [item.strip() for item in str(args.subsets).split(",") if item.strip()]

    manifest: dict[str, Any] = {
        "args": vars(args),
        "run_name": run_name,
        "designs": {int(d): DESIGN_NAMES[d] for d in sorted(set(designs))},
        "git": _git_commit(),
        "model_ckpt": model_ckpt,
        "model_config": str(engine.model_config),
        "load_info": engine.load_info,
        "device": str(engine.device),
        "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else "",
        "host": platform.node(),
        "clips": [],
    }
    manifest_path = run_dir / "manifest.json"

    def _flush_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _flush_manifest()
    run_start = time.perf_counter()
    for subset in subsets:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            logger.warning("Skipping missing subset directory: %s", subset_dir)
            continue
        seq_paths = sorted(subset_dir.glob("*.npz"))
        if int(args.limit_seqs) > 0:
            seq_paths = seq_paths[: int(args.limit_seqs)]
        logger.info("Dumping subset=%s sequences=%d clip_starts=%s designs=%s", subset, len(seq_paths), clip_starts, designs)
        for seq_path in seq_paths:
            for clip_start in clip_starts:
                out_path = run_dir / subset / f"{seq_path.stem}__c{int(clip_start)}.npz"
                entry: dict[str, Any] = {
                    "subset": subset,
                    "sequence": seq_path.stem,
                    "clip_start": int(clip_start),
                    "file": str(out_path.relative_to(run_dir)),
                }
                if out_path.exists() and not args.overwrite:
                    entry["status"] = "skipped_exists"
                    manifest["clips"].append(entry)
                    logger.info("Skipping existing %s", out_path)
                    continue

                t0 = time.perf_counter()
                try:
                    clip = load_clip(seq_path, clip_start=int(clip_start), clip_len=int(args.clip_len), subset=subset)
                except RuntimeError as exc:
                    entry["status"] = f"skipped: {exc}"
                    manifest["clips"].append(entry)
                    logger.warning("Skipping %s c%d: %s", seq_path.name, clip_start, exc)
                    continue
                load_s = time.perf_counter() - t0

                memory_path = run_dir / subset / f"{seq_path.stem}__c{int(clip_start)}.memory.pt" if args.save_memory else None
                engine.encode(clip.video, aspect_wh=clip.aspect_wh, save_memory_path=memory_path)
                encode_s = engine.last_encode_seconds

                queries, pred, decode_info = decode_designs(engine, clip, designs, seed=int(args.seed))
                decode_s = decode_info["decode_s"]
                gt = fill_gt(clip, queries)
                table = build_table(
                    clip,
                    queries,
                    pred,
                    gt,
                    model_ckpt=model_ckpt,
                    model_config=str(engine.model_config),
                    memory_path=str(memory_path) if memory_path is not None else None,
                )
                validate_table(table)
                t1 = time.perf_counter()
                _write_npz(out_path, table)
                write_s = time.perf_counter() - t1

                counts = {DESIGN_NAMES[d]: int(np.count_nonzero(queries["design"] == d)) for d in sorted(set(designs))}
                entry.update(
                    {
                        "status": "ok",
                        "T": int(clip.T),
                        "N": int(clip.N),
                        "hw_native": list(clip.hw_native),
                        "has_pose_gt": bool(clip.has_pose_gt),
                        "fixed_camera": bool(clip.fixed_camera),
                        "seq_frames": int(clip.extras["seq_frames"]),
                        "median_depth": float(clip.median_depth),
                        "vis_inside_frac": float(clip.extras["vis_inside_frac"]),
                        "num_static": int(clip.is_static.sum()),
                        "num_queries": int(queries["design"].shape[0]),
                        "queries_per_design": counts,
                        "load_s": float(load_s),
                        "encode_s": float(encode_s),
                        "decode_s": float(decode_s),
                        "write_s": float(write_s),
                        **{k: v for k, v in decode_info.items() if k != "decode_s"},
                        "file_bytes": int(out_path.stat().st_size),
                        "memory_path": str(memory_path) if memory_path is not None else "",
                    }
                )
                entry.update(clip_sanity(table))
                manifest["clips"].append(entry)
                _flush_manifest()
                msg = (
                    f"[dump] {subset}/{seq_path.stem} c{int(clip_start)} T={clip.T} N={clip.N} "
                    f"queries={entry['num_queries']} {counts} encode={encode_s:.2f}s decode={decode_s:.2f}s "
                    f"load={load_s:.2f}s write={write_s:.2f}s size={entry['file_bytes'] / 1e6:.1f}MB "
                    f"s={entry['scale_selfframe']:.4g} d0_apd={entry.get('d0_apd_global', float('nan')):.4f}"
                )
                logger.info(msg)
                engine.clear()

    manifest["total_seconds"] = float(time.perf_counter() - run_start)
    _flush_manifest()
    logger.info("Wrote %d clip entries; manifest %s", len(manifest["clips"]), manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
