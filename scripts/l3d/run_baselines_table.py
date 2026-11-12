#!/usr/bin/env python
"""WorldTrack baseline table: OpenD4RT vs ARROW vs TrackEverything under both query rules.

Reads the ``summary.json`` files that ``score_worldtrack_preds.py`` wrote for each model's saved
predictions (all queries = the upstream Gate 0 rule; ``--in-image`` = ARROW's rule), ARROW's own
official ``summary.json`` as a check, and the per-sequence ``seconds`` saved with the ARROW and
TrackEverything predictions. Writes ``results/l3d/baselines/baselines.{json,md}``. CPU only.

    python scripts/l3d/run_baselines_table.py --root output/l3d/baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d.report import fmt, jsonable  # noqa: E402

SUBSETS = ("adt_mini", "po_mini", "pstudio_mini", "ds_mini")
# model -> (scored dir, all queries; scored dir, in-image; prediction dir with per-sequence seconds)
MODELS = {
    "OpenD4RT": ("opend4rt_scored", "opend4rt_scored_inimage", None),
    "ARROW": ("arrow/ourqueries_scored", "arrow/ourqueries_scored_inimage", "arrow/ourqueries_preds"),
    "TrackEverything (Pi3)": ("trackeverything_scored", "trackeverything_scored_inimage", "trackeverything"),
}
METRICS = {"apd": "avg_pts_global", "epe": "epe_global", "apd_dyn": "avg_pts_global_dyn", "epe_dyn": "epe_global_dyn"}
RULES = ("in_image", "all")


def load_scored(path: Path) -> dict[str, dict[str, Any]]:
    """Per-subset metrics, query counts and sequence count of one ``score_worldtrack_preds`` run."""
    subsets = json.loads((path / "summary.json").read_text())["subsets"]
    out = {}
    for s in SUBSETS:
        d = subsets[s]
        if d.get("missing_predictions"):
            raise ValueError(f"{path}/{s}: missing predictions {d['missing_predictions'][:3]}")
        out[s] = {k: float(d[v]) for k, v in METRICS.items()}
        out[s].update(queries=int(d["total_queries"]), sequences=len(d["sequences"]),
                      off_image_dropped=int(d.get("off_image_queries_dropped", 0)))
    return out


def seconds(pred_dir: Path) -> dict[str, float]:
    """Mean model seconds per sequence from the ``seconds`` field of each prediction file."""
    out = {}
    for s in SUBSETS:
        vals = []
        for f in sorted((pred_dir / s).glob("*.npz")):
            with np.load(f) as z:
                if "seconds" in z.files:
                    vals.append(float(z["seconds"]))
        out[s] = float(np.mean(vals)) if vals else float("nan")
    return out


def mean_row(per_subset: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {k: float(np.mean([per_subset[s][k] for s in SUBSETS])) for k in METRICS}


def pair_table(res: dict[str, Any], rule: str, a: str, b: str) -> list[str]:
    names = list(MODELS)
    lines = ["| Subset | " + " | ".join(names) + " | Queries |", "| --- | " + " | ".join("---:" for _ in names) + " | ---: |"]
    for s in (*SUBSETS, "mean"):
        cells = []
        for m in names:
            row = res[m][rule]["mean"] if s == "mean" else res[m][rule]["subsets"][s]
            cells.append(f"{fmt(row[a])} / {fmt(row[b])}")
        q = res[names[0]][rule]["subsets"][s] if s != "mean" else None
        qcell = "" if q is None else (f"{q['queries']}" + (f" ({q['off_image_dropped']} dropped)" if rule == "in_image" else ""))
        label = "**mean**" if s == "mean" else f"`{s}`"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {qcell} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "output/l3d/baselines")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/baselines")
    args = ap.parse_args()

    res: dict[str, Any] = {}
    for m, (all_dir, in_dir, pred_dir) in MODELS.items():
        res[m] = {}
        for rule, d in (("all", all_dir), ("in_image", in_dir)):
            per = load_scored(args.root / d)
            res[m][rule] = {"subsets": per, "mean": mean_row(per)}
        res[m]["seconds_per_sequence"] = seconds(args.root / pred_dir) if pred_dir else None
    for rule in RULES:  # every model must be scored on the same query set
        for s in SUBSETS:
            counts = {m: res[m][rule]["subsets"][s]["queries"] for m in MODELS}
            if len(set(counts.values())) != 1:
                raise ValueError(f"{rule}/{s}: query counts differ {counts}")

    official = json.loads((args.root / "arrow/results/worldtrack/summary.json").read_text())["datasets"]
    check = {s: {"official_apd": float(official[s]["metrics"]["APD"]),
                 "ours_in_image_apd": res["ARROW"]["in_image"]["subsets"][s]["apd"],
                 "official_epe": float(official[s]["metrics"]["EPE"]),
                 "ours_in_image_epe": res["ARROW"]["in_image"]["subsets"][s]["epe"]} for s in SUBSETS}
    max_diff = max(abs(c["official_apd"] - c["ours_in_image_apd"]) for c in check.values())
    out = {"models": res, "arrow_official_check": check, "arrow_official_max_apd_diff": max_diff,
           "arrow_paper_mean_apd": {"ARROW": 0.8814, "OpenD4RT": 0.7378}}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baselines.json").write_text(json.dumps(jsonable(out), indent=1) + "\n")

    secs = {m: res[m]["seconds_per_sequence"] for m in MODELS if res[m]["seconds_per_sequence"]}
    md = [
        "# WorldTrack baselines: OpenD4RT vs ARROW vs TrackEverything",
        "",
        "Run 2026-11-11/12 on the AWS g6e.xlarge (1x L40S). Every number comes from "
        "`scripts/l3d/score_worldtrack_preds.py`, which applies the upstream Gate 0 metric code "
        "(`_metrics_for_sequence`, `_aggregate_results` of `eval_track3d_in_worldtrack.py`) to each model's saved "
        "predictions: first 64 frames, frame-0 queries, camera-0 coordinates, one global median scale per sequence, "
        "APD over {0.1, 0.3, 0.5, 1.0}, means over sequences. All three models are scored on the same query set "
        "(checked). Cells are **APD / EPE**; higher APD and lower EPE are better. Regenerate with "
        "`python scripts/l3d/run_baselines_table.py`.",
        "",
        "## In-image queries (ARROW's rule)",
        "",
        "Frame-0 visible queries whose pixel lies inside the image (`-0.5 <= u <= W - 0.5`, same for v); "
        "`score_worldtrack_preds.py --in-image`.",
        "",
        *pair_table(res, "in_image", "apd", "epe"),
        "",
        "## All queries (upstream Gate 0 rule)",
        "",
        "Frame-0 visible queries with a finite pixel and depth, no image bound. OpenD4RT reproduces Gate 0 exactly.",
        "",
        *pair_table(res, "all", "apd", "epe"),
        "",
        "## Dynamic points, in-image queries",
        "",
        "Cells are **APD dyn / EPE dyn** (points whose total GT motion exceeds 0.01; scale refit on them).",
        "",
        *pair_table(res, "in_image", "apd_dyn", "epe_dyn"),
        "",
        "## Check against ARROW's own evaluation",
        "",
        "| Subset | ARROW official APD | Ours, in-image | ARROW official EPE | Ours, in-image |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| `{s}` | {fmt(c['official_apd'], 4)} | {fmt(c['ours_in_image_apd'], 4)} | {fmt(c['official_epe'], 4)} | "
          f"{fmt(c['ours_in_image_epe'], 4)} |" for s, c in check.items()],
        "",
        f"Largest APD difference: {max_diff:.1e}. ARROW's official 4-subset mean APD is "
        f"{np.mean([c['official_apd'] for c in check.values()]):.4f}, as in its paper (0.8814).",
        "",
        "## Notes",
        "",
        "- **OpenD4RT**: `OpenD4RT_48CLIP_9Mix_NoCropAUG` through the unmodified upstream eval, with "
        "`scripts/l3d/run_opend4rt_worldtrack_preds.py` saving the predictions it scores. The rerun's upstream summary "
        "is identical to Gate 0 (`results/l3d/gate0/`).",
        "- **ARROW** (github.com/ifradlin/arrow at e0e1a76, weights `ifradlin/arrow`): inputs exactly as its own "
        "WorldTrack task builds them (max side 504), queried with our set by "
        "`scripts/l3d/run_arrow_worldtrack_ourqueries.py`.",
        "- **TrackEverything** (github.com/ayushjain1144/trackeverything at dfb29b5, weights "
        "`katefgroup/TrackEverything`, sparse mode) via `scripts/l3d/run_trackeverything_worldtrack.py`, with **Pi3** "
        "geometry: the default VGGT-Omega checkpoint is a private cluster path. These numbers probably understate the "
        "paper's setup.",
        "- **The query rule only matters on ds**: 13.9% of its queries project off-image (some at |u| ~ 1e6 px, from "
        "near-zero depth); adt, po and pstudio lose 19, 10 and 14 queries. Dropping them raises ds APD by 0.04-0.07 for "
        "all three models.",
        "- ARROW's paper lists OpenD4RT at 0.7378 mean APD; we get "
        f"{res['OpenD4RT']['all']['mean']['apd']:.4f} with all queries and {res['OpenD4RT']['in_image']['mean']['apd']:.4f} "
        "in-image. The query rule explains part of the gap but overshoots; the rest is unexplained (possibly how ARROW "
        "ran OpenD4RT).",
        "- ARROW's own EPE dyn divides by every sequence (a sequence with no dynamic points counts as 0); ours averages "
        "only over sequences that have dynamic points, so ARROW's own figure is lower. Compare EPE dyn within one "
        "scorer only.",
        "- Every pstudio point is dynamic, so its dyn numbers equal the overall ones. One adt sequence "
        "(`Apartment_release_decoration_skeleton_seq139_8`) has no frame-0 queries; 199 sequences are scored.",
        "- Mean model seconds per sequence (inference only, L40S): "
        + "; ".join(f"{m} {np.nanmean(list(v.values())):.1f} s" for m, v in secs.items())
        + "; OpenD4RT's whole upstream eval took 46 min for 199 sequences (about 14 s each, including loading). "
        "TrackEverything's Pi3 ran on a slow PyTorch RoPE2D fallback.",
        "- Raw predictions and per-sequence scores: `output/l3d/baselines/` (not committed).",
        "",
    ]
    (args.out_dir / "baselines.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
