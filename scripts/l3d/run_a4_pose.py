#!/usr/bin/env python
"""A4: pose baselines from the decoded grid (design 2); writes a4_pose.{json,md}.

Per clip, :func:`l3d.check_rows.pose_baselines` estimates ``T_{t->c}`` for every grid
pair by Umeyama, confidence-weighted Umeyama and ray center (label-free K, and GT K),
from outputs scaled by the clip scale. Per subset: medians over all pairs of the rotation
error and the translation error relative to the clip's median GT depth, the GT motion
for reference, ATE / RPE of the ``T_{t->0}`` trajectories (median over clips), and the
per-pair median camera-swap and ray-center residuals (Gate 1 (iv) compares them with the
pose error).

Example:
    .venv/bin/python scripts/l3d/run_a4_pose.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import check_rows as cr  # noqa: E402
from l3d.decompose import table_scale  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable, model_label  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
LABEL = {"umeyama": "Umeyama", "umeyama_conf": "Umeyama, conf-weighted", "ray_center": "ray center (K fit)",
         "ray_center_gtK": "ray center (GT K)"}


def pool(clips: list[dict[str, Any]]) -> dict[str, Any]:
    cat = lambda f: np.concatenate([f(c) for c in clips]) if clips else np.zeros(0)  # noqa: E731
    pairs = {k: cat(lambda c, k=k: c["pairs"][k]) for k in ("gt_rot_deg", "gt_trans_rel", "cs_residual", "rc_spread")}
    out: dict[str, Any] = {
        "n_clips": len(clips),
        "n_pairs": int(pairs["gt_rot_deg"].size),
        "gt_motion": {"rot_deg_median": float(np.median(pairs["gt_rot_deg"])),
                      "trans_rel_median": float(np.median(pairs["gt_trans_rel"]))},
        "cs_residual_median": float(np.median(pairs["cs_residual"])),
        "rc_spread_median": float(np.median(pairs["rc_spread"])),
        "methods": {},
    }
    base_t = cat(lambda c: c["methods"]["umeyama"]["trans_rel"])
    for m in cr.POSE_METHODS:
        rot = cat(lambda c, m=m: c["methods"][m]["rot_deg"])
        tr = cat(lambda c, m=m: c["methods"][m]["trans_rel"])
        traj = [c["trajectory"][m] for c in clips if m in c["trajectory"]]
        out["methods"][m] = {
            "rot_deg_median": float(np.median(rot)), "rot_deg_mean": float(np.mean(rot)),
            "trans_rel_median": float(np.median(tr)), "trans_rel_mean": float(np.mean(tr)),
            "beats_umeyama_trans": float(np.mean(tr < base_t)) if m != "umeyama" else float("nan"),
            "n_traj": len(traj),
            **({k: float(np.median([t[k] for t in traj])) for k in ("ate_rel", "rpe_trans_rel", "rpe_rot_deg")}
               if traj else {}),
        }
    return out


def write_md(per_subset: dict[str, dict[str, Any]], k_ratio: dict[str, list[float]], path: Path, argv: list[str],
             runtime_s: float, label: str = "OpenD4RT 48CLIP") -> None:
    lines = [
        f"# A4: pose baselines from the decoded grid ({label}, design 2, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). Per grid pair `(t, c)`: a 32x32 pixel grid of frame t decoded in",
        "cameras t and c (`c` in {0, t-4, t+4}), outputs scaled by the clip scale, `T_{t->c}` fitted by each method and",
        "compared with the GT relative pose. Translation errors are relative to the clip's median GT depth. ATE / RPE:",
        "trajectory `T_{t->0}` vs GT after a Sim(3) alignment, median over clips (not computed for static cameras).",
        "",
        "Label-free K on the grid: fx(fit) / fx(GT), median over clips: "
        + ", ".join(f"`{s}` {fmt(float(np.median(v)), 3)}" for s, v in k_ratio.items() if v) + ".",
        "",
        "## Relative pose per pair (median over pairs)",
        "",
        "| Subset | Pairs | GT rot (deg) | GT trans / depth | Method | rot err (deg) | trans err / depth | beats Umeyama (trans) "
        "| ATE / depth | RPE trans / depth | RPE rot (deg) |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r in per_subset.items():
        if not r.get("n_clips"):
            continue
        for k, m in enumerate(cr.POSE_METHODS):
            v = r["methods"][m]
            head = (f"| `{s}` | {r['n_pairs']} | {fmt(r['gt_motion']['rot_deg_median'], 2)} | "
                    f"{fmt(r['gt_motion']['trans_rel_median'], 4)} " if k == 0 else "| | | | ")
            lines.append(head + f"| {LABEL[m]} | {fmt(v['rot_deg_median'], 3)} | {fmt(v['trans_rel_median'], 4)} | "
                         f"{fmt(v['beats_umeyama_trans'], 2)} | {fmt(v.get('ate_rel'), 4)} | "
                         f"{fmt(v.get('rpe_trans_rel'), 4)} | {fmt(v.get('rpe_rot_deg'), 3)} |")
    lines += [
        "",
        "## Label-free camera residuals vs pose error (Gate 1 (iv): median residual >= 20% of the median pose error)",
        "",
        "| Subset | median camera-swap residual (SE(3)) | median ray-center spread | median Umeyama trans err / depth "
        "| camera swap / pose err | ray center / pose err |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r in per_subset.items():
        if r.get("n_clips"):
            pe = r["methods"]["umeyama"]["trans_rel_median"]
            lines.append(f"| `{s}` | {fmt(r['cs_residual_median'], 4)} | {fmt(r['rc_spread_median'], 4)} | {fmt(pe, 4)} | "
                         f"{fmt(r['cs_residual_median'] / pe if pe > 0 else float('nan'), 2)} | "
                         f"{fmt(r['rc_spread_median'] / pe if pe > 0 else float('nan'), 2)} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--model-label", default=None, help="model name in the report (default: from the dump manifest)")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/a4_pose")
    args = ap.parse_args()

    subsets = args.subsets.split(",")
    start = time.time()
    clips: dict[str, list[dict[str, Any]]] = {s: [] for s in subsets}
    k_ratio: dict[str, list[float]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        pb = cr.pose_baselines(d, cr.row_errors(d, table_scale(d)))
        clips[entry["subset"]].append(pb)
        k_ratio[entry["subset"]].append(float(pb["K_fit"][0, 0] / d["K_gt"][0, 0]))
        print(f"[a4] {entry['file']}", flush=True)
    per_subset = {s: pool(c) if c else {"n_clips": 0} for s, c in clips.items()}
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "per_subset": per_subset, "k_fit_ratio": k_ratio,
               "runtime_s": runtime}
    (args.out / "a4_pose.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, k_ratio, args.out / "a4_pose.md", sys.argv, runtime, label=model_label(args.dump, args.model_label))
    print((args.out / "a4_pose.md").read_text())


if __name__ == "__main__":
    main()
