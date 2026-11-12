#!/usr/bin/env python
"""Part A split by GT visibility, motion and parallax; writes results/l3d/splits/splits.{json,md}.

Per clip, :func:`l3d.splits.clip_sums` labels design-0 rows and design-1 cells and sums,
per split and bin: tracking APD/EPE (design 0 in camera 0), the self-frame depth share and
relative error (design 0, ``t_c = t_t``), and the T/S/C parts of the 3D error (complete
design-1 cells). Sums are pooled over a subset's clips, as in A1 and A3. CPU only.

Example:
    .venv/bin/python scripts/l3d/run_splits.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import splits  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable, model_label  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
TITLES = {
    "visibility": "Visibility (GT, at the target frame)",
    "motion": "Motion (GT static: world displacement over the clip < 1% of median depth)",
    "parallax": "Parallax (angle at the GT point between the source and target cameras, degrees)",
}


def write_md(res: dict[str, Any], path: Path, label: str, argv: list[str], runtime_s: float) -> None:
    lines = [
        f"# Part A by visibility, motion and parallax ({label}, clip [0, 48))",
        "",
        f"`{' '.join(['scripts/l3d/run_splits.py', *argv])}` ({runtime_s:.0f} s; code `l3d/splits.py`). Sums are pooled over "
        "each subset's clips. Columns:",
        "",
        "- **Rows**: design-0 rows (frame-0 queries, every target frame) in the bin, and their share of the subset.",
        "- **APD / EPE**: tracking accuracy of those rows decoded in camera 0 (the upstream protocol inside the clip), "
        "with the clip's global median scale; pooled over rows, so not the upstream per-sequence mean.",
        "- **Rel. err / depth share**: self-frame (`t_c = t_t`) error |P - X| / |X| and the share of its squared error "
        "along the ray (A1's depth share).",
        "- **Cells, T / S / C / SC**: complete design-1 cells (one point and target frame, crossed source x camera "
        "frames) in the bin and the T/S/C shares of their 3D error (A3); `1 - T` is the disagreement.",
        "- Parallax of a design-0 row uses `t_s = 0`, so it grows with the frame gap; a design-1 cell uses its sources' "
        "median. pstudio has a fixed camera, so all of it is in the first parallax bin.",
    ]
    for split, title in TITLES.items():
        lines += ["", f"## {title}", "",
                  "| Subset | Bin | Rows | Share | APD | EPE | Rel. err | Depth share | Cells | T | S | C | SC |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for subset in SUBSETS:
            for b in res["subsets"][subset]["summary"][split]:
                if b["track_rows"] == 0 and b["tsc_cells"] == 0:
                    lines.append(f"| `{subset}` | {b['bin']} | 0 | | | | | | 0 | | | | |")
                    continue
                lines.append(
                    f"| `{subset}` | {b['bin']} | {b['track_rows']} | {fmt(b['track_frac'], 2)} | {fmt(b['apd'])} | "
                    f"{fmt(b['epe'])} | {fmt(b['self_rel_err'])} | {fmt(b['depth_share'], 2)} | {b['tsc_cells']} | "
                    f"{fmt(b['T_share'], 2)} | {fmt(b['S_share'], 2)} | {fmt(b['C_share'], 2)} | {fmt(b['SC_share'], 2)} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/splits")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--model-label", default=None, help="model name in the report (default: from the dump manifest)")
    args = ap.parse_args()

    start = time.time()
    subsets = args.subsets.split(",")
    per_clip: dict[str, list] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        per_clip[entry["subset"]].append(splits.clip_sums(d))
    res: dict[str, Any] = {"dump": str(args.dump.relative_to(REPO) if args.dump.is_relative_to(REPO) else args.dump),
                           "parallax_edges_deg": list(splits.PARALLAX_EDGES), "subsets": {}}
    for s in subsets:
        pooled = splits.pool(per_clip[s])
        res["subsets"][s] = {"n_clips": len(per_clip[s]), "summary": splits.summarize(pooled), "sums": pooled}
    runtime = time.time() - start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits.json").write_text(json.dumps(jsonable(res), indent=1) + "\n")
    write_md(res, args.out_dir / "splits.md", model_label(args.dump, args.model_label), sys.argv[1:], runtime)
    print((args.out_dir / "splits.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
