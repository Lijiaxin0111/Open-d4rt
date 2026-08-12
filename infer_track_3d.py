"""Minimal D4RT inference helpers used by WorldTrack evaluation."""

from __future__ import annotations

from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from src.eval.tasks import _encode_model_memory, _model_clip_frames, _run_model_for_queries, _umeyama_sim3


def _resolve_device(raw: str) -> torch.device:
    key = raw.strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


def _resize_video(video_rgb: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    resized = [
        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR)
        for frame in video_rgb
    ]
    return np.stack(resized, axis=0)


def _load_video_rgb(path: Path, max_frames: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 10.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames > 0 and len(frames) >= int(max_frames):
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def _grid_query_points(
    width: int,
    height: int,
    cols: int,
    rows: int,
    margin_ratio: float,
    max_points: int,
) -> np.ndarray:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    margin_x = float(max(width - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    margin_y = float(max(height - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    xs = np.linspace(margin_x, float(max(width - 1, 0)) - margin_x, num=cols, dtype=np.float32)
    ys = np.linspace(margin_y, float(max(height - 1, 0)) - margin_y, num=rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] > max_points:
        pick = np.linspace(0, grid.shape[0] - 1, num=max_points, dtype=np.int64)
        grid = grid[pick]
    return grid.astype(np.float32)


def _make_anchor_clip_indices(num_frames: int, clip_frames: int, target_idx: int, source_idx: int = 0) -> np.ndarray:
    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    target_idx = int(np.clip(int(target_idx), 0, max(0, num_frames - 1)))
    source_idx = int(np.clip(int(source_idx), 0, max(0, num_frames - 1)))
    if num_frames <= clip_frames:
        return np.arange(num_frames, dtype=np.int64)
    if clip_frames == 1:
        return np.asarray([target_idx], dtype=np.int64)
    if source_idx != 0:
        window = [int(source_idx)]
        tail_len = clip_frames - 1
        seg_end = target_idx + 1
        seg_start = max(0, seg_end - tail_len)
        seg_end = min(num_frames, seg_start + tail_len)
        seg_start = max(0, seg_end - tail_len)
        for frame_idx in range(seg_start, seg_end):
            if frame_idx != source_idx:
                window.append(int(frame_idx))
        if target_idx not in window:
            window.append(int(target_idx))
        window = sorted(set(window))
        if len(window) > clip_frames:
            mandatory = {int(source_idx), int(target_idx)}
            ranked = sorted(
                [idx for idx in window if idx not in mandatory],
                key=lambda idx: (min(abs(idx - target_idx), abs(idx - source_idx)), idx),
            )
            keep = set(ranked[: max(0, clip_frames - len(mandatory))]) | mandatory
            window = sorted(keep)
        return np.asarray(window, dtype=np.int64)

    tail_len = clip_frames - 1
    seg_end = target_idx + 1
    seg_start = max(1, seg_end - tail_len)
    seg_end = min(num_frames, seg_start + tail_len)
    seg_start = max(1, seg_end - tail_len)
    tail = np.arange(seg_start, seg_end, dtype=np.int64)
    if target_idx not in tail:
        tail[-1] = target_idx
        tail = np.unique(tail)
        while tail.shape[0] < tail_len:
            cand = max(1, int(tail[0]) - 1)
            if cand in tail:
                break
            tail = np.concatenate([np.asarray([cand], dtype=np.int64), tail], axis=0)
        tail = np.sort(tail)[-tail_len:]
    return np.concatenate([np.asarray([0], dtype=np.int64), tail], axis=0)


def _make_sliding_window_clip_ranges(
    num_frames: int,
    clip_frames: int,
    overlap_frames: int | None = None,
) -> list[tuple[int, int]]:
    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    if num_frames <= clip_frames:
        return [(0, num_frames)]
    overlap = clip_frames // 2 if overlap_frames is None else int(overlap_frames)
    overlap = max(1, min(overlap, clip_frames - 1))
    stride = max(1, clip_frames - overlap)
    starts = list(range(0, max(num_frames - clip_frames, 0) + 1, stride))
    last_start = max(0, num_frames - clip_frames)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(int(start), int(min(num_frames, start + clip_frames))) for start in starts]


def _apply_sim3_to_xyz(
    xyz: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    src = np.asarray(xyz, dtype=np.float64)
    out = np.full_like(src, np.nan, dtype=np.float64)
    flat_src = src.reshape(-1, 3)
    flat_out = out.reshape(-1, 3)
    valid = np.isfinite(flat_src).all(axis=1)
    if np.any(valid):
        flat_out[valid] = (float(scale) * (np.asarray(rot, dtype=np.float64) @ flat_src[valid].T)).T + np.asarray(
            trans, dtype=np.float64
        )
    return out.astype(np.float32)


def _estimate_overlap_sim3(
    *,
    prev_xyz_qt3: np.ndarray,
    curr_xyz_qt3: np.ndarray,
    prev_vis_qt: np.ndarray,
    curr_vis_qt: np.ndarray,
    prev_conf_qt: np.ndarray,
    curr_conf_qt: np.ndarray,
    keep_ratio: float = 0.85,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    prev_xyz = np.asarray(prev_xyz_qt3, dtype=np.float64)
    curr_xyz = np.asarray(curr_xyz_qt3, dtype=np.float64)
    prev_vis = np.asarray(prev_vis_qt, dtype=bool)
    curr_vis = np.asarray(curr_vis_qt, dtype=bool)
    prev_conf = np.asarray(prev_conf_qt, dtype=np.float64)
    curr_conf = np.asarray(curr_conf_qt, dtype=np.float64)

    valid = (
        np.isfinite(prev_xyz).all(axis=-1)
        & np.isfinite(curr_xyz).all(axis=-1)
        & prev_vis
        & curr_vis
        & np.isfinite(prev_conf)
        & np.isfinite(curr_conf)
    )
    if int(np.count_nonzero(valid)) < 3:
        return None
    src = curr_xyz[valid]
    dst = prev_xyz[valid]
    scores = np.minimum(prev_conf[valid], curr_conf[valid])
    if scores.size >= 4:
        order = np.argsort(scores)[::-1]
        keep = max(3, int(np.ceil(float(scores.size) * float(np.clip(keep_ratio, 0.0, 1.0)))))
        pick = order[:keep]
        src = src[pick]
        dst = dst[pick]
    sim3 = _umeyama_sim3(src, dst)
    if sim3 is None:
        return None
    scale, rot, trans = sim3
    return float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)


def _build_query_for_targets(
    query_uv_norm: np.ndarray,
    t_src: np.ndarray,
    t_tgt: np.ndarray,
    t_cam: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "u": torch.from_numpy(query_uv_norm[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(query_uv_norm[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t_src).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t_tgt).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=device, dtype=torch.long),
    }


def _run_full_clip_queries(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_src_indices: np.ndarray | None,
    query_chunk_size: int,
    num_frames: int,
) -> dict[str, np.ndarray]:
    num_queries = int(query_uv_norm.shape[0])
    repeated_uv = np.repeat(query_uv_norm, num_frames, axis=0)
    if query_src_indices is None:
        query_src = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src = np.asarray(query_src_indices, dtype=np.int64).reshape(-1)
        if query_src.shape[0] != num_queries:
            raise ValueError(f"query_src_indices must have shape [{num_queries}], got {query_src.shape}")

    t_src = np.repeat(query_src, num_frames, axis=0)
    t_tgt = np.tile(np.arange(num_frames, dtype=np.int64), num_queries)
    memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_ratio)
    # A single ref0 pass (t_cam == 0) provides world-frame xyz plus the uv /
    # visibility / confidence heads. The former local-frame pass (t_cam == t_tgt)
    # only differed in the camera frame of its xyz output, which is never consumed.
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )

    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_frames)
            else:
                out[key] = arr.reshape(num_queries, num_frames, *arr.shape[1:])
        return out

    return _reshape(pred_ref)


def _run_clip_queries_for_target_indices(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_src_indices: np.ndarray | None,
    local_target_indices: np.ndarray,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
    target_ids = np.asarray(local_target_indices, dtype=np.int64).reshape(-1)
    num_queries = int(query_uv_norm.shape[0])
    num_targets = int(target_ids.shape[0])
    if num_targets <= 0:
        return {}

    repeated_uv = np.repeat(query_uv_norm, num_targets, axis=0)
    if query_src_indices is None:
        query_src = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src = np.asarray(query_src_indices, dtype=np.int64).reshape(-1)
        if query_src.shape[0] != num_queries:
            raise ValueError(f"query_src_indices must have shape [{num_queries}], got {query_src.shape}")

    t_src = np.repeat(query_src, num_targets)
    t_tgt = np.tile(target_ids, num_queries)
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )

    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_targets)
            else:
                out[key] = arr.reshape(num_queries, num_targets, *arr.shape[1:])
        return out

    return _reshape(pred_ref)


def _infer_tracks(
    *,
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    native_aspect_ratio: float,    
    query_uv_norm: np.ndarray,
    query_chunk_size: int,
    query_src_indices_global: np.ndarray | None = None,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
    dense_grid_size: int = 32,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    num_queries = int(query_uv_norm.shape[0])
    clip_frames = _model_clip_frames(model)
    aspect_value = np.asarray([[native_aspect_ratio]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)

    tracks_xyz_ref0 = np.full((num_queries, num_frames, 3), np.nan, dtype=np.float32)
    tracks_uv = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    tracks_visibility = np.zeros((num_queries, num_frames), dtype=bool)
    tracks_visibility_logits = np.full((num_queries, num_frames), np.nan, dtype=np.float32)
    tracks_confidence = np.full((num_queries, num_frames), np.nan, dtype=np.float32)
    dense_mode = bool(umeyama_slide_window_dense)
    slide_window_enabled = bool(umeyama_slide_window) or dense_mode
    stitch_diagnostics: dict[str, Any] = {
        "mode": "umeyama_slide_window_dense" if dense_mode else ("umeyama_slide_window" if slide_window_enabled else "anchor_clip"),
        "clip_frames": int(clip_frames),
        "dense_grid_size": int(dense_grid_size) if dense_mode else 0,
        "keep_ratio": 0.85,
        "chunks": [],
    }
    if query_src_indices_global is None:
        query_src_indices_global = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src_indices_global = np.asarray(query_src_indices_global, dtype=np.int64).reshape(num_queries)

    video_tensor = (
        torch.from_numpy(video_model_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )

    with torch.no_grad():
        if num_frames <= clip_frames:
            pred_ref = _run_full_clip_queries(
                model=model,
                video_clip=video_tensor,
                aspect_ratio=aspect_tensor,
                query_uv_norm=query_uv_norm,
                query_src_indices=query_src_indices_global,
                query_chunk_size=query_chunk_size,
                num_frames=num_frames,
            )
            tracks_xyz_ref0[:] = pred_ref["xyz_3d"].astype(np.float32)
            tracks_uv[:] = pred_ref["uv_2d"].astype(np.float32)
            tracks_visibility_logits[:] = pred_ref["visibility"].astype(np.float32)
            tracks_visibility[:] = 1.0 / (1.0 + np.exp(-tracks_visibility_logits)) > 0.5
            tracks_confidence[:] = pred_ref["confidence"].astype(np.float32)
        else:
            clip_groups: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
            for frame_idx in range(num_frames):
                for src_idx in sorted(set(int(v) for v in query_src_indices_global.tolist())):
                    clip_indices = _make_anchor_clip_indices(
                        num_frames=num_frames,
                        clip_frames=clip_frames,
                        target_idx=frame_idx,
                        source_idx=src_idx,
                    )
                    local_tgt_idx = int(np.flatnonzero(clip_indices == frame_idx)[0])
                    local_src_idx = int(np.flatnonzero(clip_indices == int(src_idx))[0])
                    clip_groups.setdefault(tuple(int(v) for v in clip_indices.tolist()), []).append(
                        (frame_idx, local_tgt_idx, int(src_idx), local_src_idx)
                    )

            for clip_key, assignments in clip_groups.items():
                clip_indices = np.asarray(clip_key, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)

                by_src: dict[int, list[tuple[int, int, int]]] = {}
                for frame_idx, local_tgt_idx, src_idx, local_src_idx in assignments:
                    by_src.setdefault(int(src_idx), []).append(
                        (int(frame_idx), int(local_tgt_idx), int(local_src_idx))
                    )
                for src_idx, src_assignments in by_src.items():
                    global_frame_ids = np.asarray([item[0] for item in src_assignments], dtype=np.int64)
                    local_target_ids = np.asarray([item[1] for item in src_assignments], dtype=np.int64)
                    local_src_idx = int(src_assignments[0][2])
                    valid_idx = np.flatnonzero(query_src_indices_global == int(src_idx))
                    if valid_idx.size <= 0:
                        continue
                    local_src_ids = np.full((valid_idx.shape[0],), local_src_idx, dtype=np.int64)
                    pred_ref = _run_clip_queries_for_target_indices(
                        model=model,
                        video_clip=video_clip,
                        aspect_ratio=aspect_tensor,
                        memory=memory,
                        query_uv_norm=query_uv_norm[valid_idx],
                        query_src_indices=local_src_ids,
                        local_target_indices=local_target_ids,
                        query_chunk_size=query_chunk_size,
                    )

                    tracks_xyz_ref0[valid_idx[:, None], global_frame_ids[None, :]] = pred_ref["xyz_3d"].astype(np.float32)
                    tracks_uv[valid_idx[:, None], global_frame_ids[None, :]] = pred_ref["uv_2d"].astype(np.float32)
                    pred_vis_logits = pred_ref["visibility"].astype(np.float32)
                    tracks_visibility_logits[valid_idx[:, None], global_frame_ids[None, :]] = pred_vis_logits
                    tracks_visibility[valid_idx[:, None], global_frame_ids[None, :]] = (
                        1.0 / (1.0 + np.exp(-pred_vis_logits)) > 0.5
                    )
                    tracks_confidence[valid_idx[:, None], global_frame_ids[None, :]] = pred_ref["confidence"].astype(np.float32)

    return {
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_uv_norm": tracks_uv,
        "tracks_visibility": tracks_visibility,
        "tracks_visibility_logits": tracks_visibility_logits,
        "tracks_confidence": tracks_confidence,
        "clip_frames": np.asarray(clip_frames, dtype=np.int32),
        "stitch_diagnostics": stitch_diagnostics,
    }
