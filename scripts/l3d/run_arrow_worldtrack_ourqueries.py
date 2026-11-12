"""Run ARROW on OUR WorldTrack queries and save predictions for ``score_worldtrack_preds.py``.

Our queries are the upstream Gate 0 ones (frame-0 visible, finite pixel and depth, no in-bounds
filter). Model inputs are built exactly as ARROW's ``WorldTrackTask.prepare`` (same scene loader,
max side 504, resized intrinsics, same xy formula); only the query id set differs from ARROW's
in-bounds-filtered set. Saves ``<out>/<subset>/<seq>.npz`` with ``pred_tracks_ref0 [T, N, 3]``
(camera-0 frame), ``query_uv [N, 2]`` (native pixels, float64) and ``seconds``.

Runs in ARROW's own env (github.com/ifradlin/arrow, weights ``ifradlin/arrow``, CC BY-NC), after
ARROW's WorldTrack preprocessing has written ``data/eval/worldtrack`` under its repo. From ARROW's
repo directory (on the Deep Learning AMI, clear ``LD_LIBRARY_PATH`` so the venv's cuDNN is used):
    LD_LIBRARY_PATH= uv run python /path/to/l3d-d4rt/scripts/l3d/run_arrow_worldtrack_ourqueries.py \
        /path/to/worldtrack_release /path/to/preds/arrow_ourqueries
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from arrow.eval.dataset import EvalDataset
from arrow.eval.interface import ARROWInterface
from arrow.tasks.tracking import tracking_predictions


def our_queries(npz_path):
    with np.load(npz_path, allow_pickle=True) as pack:
        xyz = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
        vis = np.asarray(pack["visibility"], dtype=bool)
        fx, fy, cx, cy = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64).reshape(-1)[:4]
    z = xyz[0, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
        uv = np.stack([xyz[0, :, 0] / safe_z * fx + cx, xyz[0, :, 1] / safe_z * fy + cy], -1)
    visible = vis[0]
    keep_all = visible & np.isfinite(uv).all(-1) & np.isfinite(z) & (np.abs(z) > 1e-8)
    ids = np.nonzero(keep_all)[0]
    return ids, uv[ids]


def main():
    raw_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_float32_matmul_precision("high")
    dataset = EvalDataset(root="data/eval/worldtrack", subsets=["adt_mini", "ds_mini", "po_mini", "pstudio_mini"],
                          max_frames=64, max_side=504)
    interface = ARROWInterface("ifradlin/arrow", device="cuda", decode_chunk_size=100_000)
    log = []
    for scene in dataset:
        out = out_dir / scene.subset / f"{scene.name}.npz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        ids, query_uv = our_queries(raw_root / scene.subset / f"{scene.name}.npz")
        if len(ids) == 0:
            print(f"[arrow-ours] skip {scene.subset}/{scene.name}: no queries", flush=True)
            continue
        tracks, visibility = scene.tracks_world, scene.visibility
        w2c = torch.linalg.inv(scene.poses)
        source = tracks[0] @ w2c[0, :3, :3].T + w2c[0, :3, 3]
        xy = source[:, :2] / source[:, 2:].clamp_min(1e-6)
        xy = xy @ scene.intrinsics[:2, :2].T + scene.intrinsics[:2, 2]
        h, w = scene.observations.images.shape[-2:]
        inb = visibility[0] & (xy >= -.5).all(-1) & (xy <= xy.new_tensor([w, h]) - .5).all(-1)
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        start = time.time()
        result = interface(observations=scene.observations, tasks=["track"], xy=xy[ids_t], source_frame=0,
                           reference_frame=0, skip_sky=False)
        points = tracking_predictions(result).points[0]  # [N, T, 3], reference frame 0
        secs = time.time() - start
        pred = points.transpose(0, 1).float().cpu().numpy()
        np.savez_compressed(out, pred_tracks_ref0=pred.astype(np.float32), query_uv=query_uv.astype(np.float64),
                            seconds=secs)
        rec = {"subset": scene.subset, "sequence": scene.name, "queries": int(len(ids)),
               "arrow_inbounds_queries": int(inb.sum()), "our_ids_outside_arrow_set": int((~inb[ids_t]).sum()),
               "frames": int(scene.observations.num_frames), "pred_shape": list(pred.shape), "seconds": secs}
        log.append(rec)
        print(f"[arrow-ours] {json.dumps(rec)}", flush=True)
        del result
    (out_dir / f"run_log_{int(time.time())}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
