#!/usr/bin/env python3
"""Run Algorithm 1 (occupancy-grid dense tracking) on a single video/clip.

This exercises ``dense_track_occupancy_grid`` from ``infer_track_3d``: it encodes
the clip once and adaptively seeds tracks only from grid cells not yet covered by
an existing track, returning a unified-frame dense 3D point set plus a speedup
report versus naive all-cell querying. No retraining is involved; it is a pure
inference-time scheduler over the frozen model.

Example:
    EXP=checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG
    python infer_dense_track.py \
        --model-config "$EXP/model.yaml" \
        --ckpt-path "$EXP/opend4rt.ckpt" \
        --input data/worldtrack_release/pstudio_mini/softball_25.npz \
        --grid 48 --seed-batch 512 --output-dir tmp/dense_track
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_track3d_in_worldtrack import load_worldtrack_sequence
from infer_track_3d import (
    _load_video_rgb,
    _resize_video,
    _resolve_device,
    _unwrap_state_dict,
    dense_track_occupancy_grid,
)
from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.eval.tasks import _model_clip_frames
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occupancy-grid dense 3D tracking (Algorithm 1).")
    parser.add_argument("--model-config", required=True, help="Model config yaml.")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path.")
    parser.add_argument("--input", required=True, help="WorldTrack .npz or a video file.")
    parser.add_argument("--output-dir", default="tmp/dense_track")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--num-frames", type=int, default=0, help="Frames to use; <=0 means the model clip length.")
    parser.add_argument("--grid", type=int, default=48, help="Occupancy-grid / seeding resolution (grid x grid).")
    parser.add_argument("--seed-batch", type=int, default=512, help="Source points seeded per iteration (batch B).")
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--mark-radius", type=int, default=1, help="Neighborhood radius when marking visited cells.")
    parser.add_argument("--visible-thresh", type=float, default=0.5)
    parser.add_argument("--reference-frame", type=int, default=0, help="Camera frame the 3D tracks are expressed in.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Safety cap on emitted tracks; <=0 disables.")
    parser.add_argument(
        "--source-frames",
        default="",
        help="Comma-separated frames allowed to seed sources. Empty means all frames.",
    )
    return parser.parse_args()


def _load_video(input_path: Path, num_frames_cap: int) -> np.ndarray:
    if input_path.suffix.lower() == ".npz":
        sample = load_worldtrack_sequence(input_path, num_frames=num_frames_cap if num_frames_cap > 0 else 1_000_000)
        return np.asarray(sample["video_rgb"])
    video_rgb, _ = _load_video_rgb(input_path, max_frames=num_frames_cap if num_frames_cap > 0 else 0)
    return np.asarray(video_rgb)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("infer_dense_track", output_dir)

    cfg = load_yaml_config(args.model_config)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg["model"]).eval()
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded checkpoint %s (missing=%d unexpected=%d)",
        ckpt_path,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    device = _resolve_device(args.device)
    model.to(device).eval()

    clip_frames = _model_clip_frames(model)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    video_rgb = _load_video(input_path, num_frames_cap=args.num_frames)
    target_frames = clip_frames if args.num_frames <= 0 else min(int(args.num_frames), clip_frames)
    if video_rgb.shape[0] > target_frames:
        logger.info("Trimming %d frames to clip length %d", int(video_rgb.shape[0]), target_frames)
        video_rgb = video_rgb[:target_frames]

    image_size = cfg.get_path("model.input.image_size", [256, 256])
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))

    source_frames = None
    if str(args.source_frames).strip():
        source_frames = [int(v) for v in str(args.source_frames).split(",") if v.strip()]

    result = dense_track_occupancy_grid(
        model=model,
        video_model_rgb=video_model_rgb,
        grid_hw=(int(args.grid), int(args.grid)),
        seed_batch_size=int(args.seed_batch),
        query_chunk_size=int(args.query_chunk_size),
        source_frames=source_frames,
        visible_thresh=float(args.visible_thresh),
        mark_radius=int(args.mark_radius),
        max_tracks=int(args.max_tracks),
        reference_frame=int(args.reference_frame),
    )
    diagnostics = result["diagnostics"]

    tracks_path = output_dir / f"{input_path.stem}_dense_tracks.npz"
    np.savez_compressed(
        tracks_path,
        tracks_xyz_ref=result["tracks_xyz_ref"],
        tracks_uv_norm=result["tracks_uv_norm"],
        tracks_visibility=result["tracks_visibility"],
        tracks_confidence=result["tracks_confidence"],
        source_uv_norm=result["source_uv_norm"],
        source_t_src=result["source_t_src"],
        reference_frame=result["reference_frame"],
        clip_frames=result["clip_frames"],
    )
    diag_path = output_dir / f"{input_path.stem}_dense_tracks_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "Dense tracking done: tracks=%d iters=%d grid=%dx%d frames=%d ref=%d",
        diagnostics["num_tracks"],
        diagnostics["num_iterations"],
        diagnostics["grid_hw"][0],
        diagnostics["grid_hw"][1],
        diagnostics["num_frames"],
        diagnostics["reference_frame"],
    )
    logger.info(
        "Adaptive speedup vs naive all-cell seeding: %.2fx (tracks %d vs %d cells); "
        "decoder-query speedup: %.2fx; grid coverage: %.1f%%",
        diagnostics["seed_speedup_ratio"],
        diagnostics["num_tracks"],
        diagnostics["naive_seed_cells"],
        diagnostics["query_speedup_ratio"],
        100.0 * diagnostics["occupancy_filled_ratio"],
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    logger.info("Saved tracks to %s", tracks_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
