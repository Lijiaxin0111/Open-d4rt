#!/usr/bin/env python
"""A3 detection: does each check residual find large errors, beyond the native confidence?

Per clip, :mod:`l3d.check_rows` gives per-row residuals, total error ``|s P - X| / |X|``,
the target component of Table 1 and ``sigmoid(conf)``; per subset the rows of all clips
are pooled and scored with :mod:`l3d.detection` (Spearman rho, AUROC for the top 20% of
errors, risk-coverage). The native confidence is scored on the same rows as
``-sigmoid(conf)``. Writes a3_detection.{json,md}.

Example:
    .venv/bin/python scripts/l3d/run_a3_detection.py
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
from l3d import detection as det  # noqa: E402
from l3d.decompose import table_scale  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
# name -> (description, target component)
CHECKS = {
    "source_swap": ("source swap, model u2 (design 5 vs 1)", "S depth: |e_d(t1) - e_d(t2)|"),
    "source_swap_gt_u2": ("source swap, GT u2 control (design 1)", "S depth: |e_d(t1) - e_d(t2)|"),
    "camera_swap_se3": ("camera swap, SE(3) fit (design 1)", "non-rigid camera-frame error"),
    "camera_swap_sim3": ("camera swap, Sim(3) fit (design 1)", "non-rigid camera-frame error"),
    "ray_center": ("ray center, K fit + R* (design 1, t_s = t_t)", "image-plane error of P(u, t, t, t)"),
    "ray_center_gt": ("ray center, GT K and pose (not label-free)", "image-plane error of P(u, t, t, t)"),
    "reprojection": ("reprojection with K fit, px (analysis only)", "image-plane error"),
    "static_rigidity": ("static rigidity, GT static mask (analysis only)", "depth error |e_d|"),
}
COVERAGE = np.linspace(0.05, 1.0, 20)


def clip_rows(d: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    err = cr.row_errors(d, table_scale(d))
    K = cr.fit_K(d, err)
    ss = cr.source_swap_detection(d, err)
    cs = cr.camera_swap_detection(d, err)
    rc = cr.ray_center_detection(d, err, K)
    base = lambda r: {k: r[k] for k in ("error", "target", "conf")}  # noqa: E731
    rows = {
        "source_swap": ss["model_u2"],
        "source_swap_gt_u2": ss["gt_u2"],
        "camera_swap_se3": {"residual": cs["residual_se3"], **base(cs)},
        "camera_swap_sim3": {"residual": cs["residual_sim3"], **base(cs)},
        "ray_center": {"residual": rc["residual"], **base(rc)},
        "ray_center_gt": {"residual": rc["residual_gt"], **base(rc)},
        "reprojection": cr.reprojection_detection(d, err, K),
        "static_rigidity": cr.static_rigidity_detection(d, err),
    }
    return {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()} for k, v in rows.items()}, float(K[0, 0] / d["K_gt"][0, 0])


def _curve(score: np.ndarray, error: np.ndarray) -> np.ndarray:
    rc = det.risk_coverage(score, error)
    if rc["risk"].size == 0:
        return np.full(COVERAGE.size, np.nan)
    idx = np.clip(np.ceil(COVERAGE * rc["risk"].size).astype(int) - 1, 0, rc["risk"].size - 1)
    return rc["risk"][idx]


def score_check(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    res, err, tgt, conf = (rows[k].astype(np.float64) for k in ("residual", "error", "target", "conf"))
    ok = np.isfinite(res) & np.isfinite(err) & np.isfinite(conf)
    res, err, tgt, conf = res[ok], err[ok], tgt[ok], conf[ok]
    if res.size < 10:
        return {"n": int(res.size)}
    cscore = -conf
    okt = np.isfinite(tgt)
    oracle = np.cumsum(np.sort(err)) / np.arange(1, err.size + 1)
    idx = np.clip(np.ceil(COVERAGE * err.size).astype(int) - 1, 0, err.size - 1)
    return {
        "n": int(res.size),
        "vs_error": {"residual": det.detection_metrics(res, err), "confidence": det.detection_metrics(cscore, err)},
        "vs_target": {
            "residual": {"spearman": det.spearman(res[okt], tgt[okt]), "auroc": det.auroc(res[okt], tgt[okt])},
            "confidence": {"spearman": det.spearman(cscore[okt], tgt[okt]), "auroc": det.auroc(cscore[okt], tgt[okt])},
            "n": int(okt.sum()),
        },
        "spearman_residual_conf": det.spearman(res, cscore),
        "median_residual": float(np.median(res)),
        "median_error": float(np.median(err)),
        "median_target": float(np.median(tgt[okt])) if okt.any() else float("nan"),
        "risk_coverage": {"coverage": COVERAGE, "residual": _curve(res, err), "confidence": _curve(cscore, err),
                          "oracle": oracle[idx]},
    }


def write_md(per_subset: dict[str, dict[str, Any]], k_ratio: dict[str, list[float]], path: Path, argv: list[str],
             runtime_s: float) -> None:
    lines = [
        "# A3: detection by each check vs native confidence (OpenD4RT 48CLIP, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). Rows of all clips of a subset are pooled. Error = `|s P - X| / |X|`",
        "of the scored query (mean over the pair for source swap); target = the component Table 1 says the check",
        "sees. Confidence is scored as `-sigmoid(conf)` on the same rows. AUROC: positives = top 20% of errors.",
        "E-AURC: excess area under the risk-coverage curve over the oracle (lower is better). Gate 1 (ii) asks for",
        "rho(residual, target) >= 0.3.",
        "",
        "Label-free K: fx(fit) / fx(GT), median over clips: "
        + ", ".join(f"`{s}` {fmt(float(np.median(v)), 3)}" for s, v in k_ratio.items() if v) + ".",
        "",
    ]
    for name, (desc, target) in CHECKS.items():
        lines += [
            f"## {desc}", "", f"Target: {target}.", "",
            "| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf "
            "| E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s, r in per_subset.items():
            m = r.get(name, {})
            if "vs_error" not in m:
                lines.append(f"| `{s}` | {m.get('n', 0)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            ve, vt = m["vs_error"], m["vs_target"]
            ratio = m["median_residual"] / m["median_error"] if m["median_error"] > 0 else float("nan")
            lines.append(
                f"| `{s}` | {m['n']} | {fmt(vt['residual']['spearman'])} | {fmt(vt['confidence']['spearman'])} | "
                f"{fmt(ve['residual']['spearman'])} | {fmt(ve['confidence']['spearman'])} | {fmt(ve['residual']['auroc'])} | "
                f"{fmt(ve['confidence']['auroc'])} | {fmt(ve['residual']['eaurc'], 4)} | {fmt(ve['confidence']['eaurc'], 4)} | "
                f"{fmt(m['spearman_residual_conf'])} | {fmt(ratio)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


PLOT_CHECKS = ("source_swap", "camera_swap_se3", "ray_center")
PLOT_LABEL = {"source_swap": "source swap (model u2)", "camera_swap_se3": "camera swap (SE(3))",
              "ray_center": "ray center (K fit)"}


def write_png(per_subset: dict[str, dict[str, Any]], path: Path) -> None:
    """Risk-coverage of each loss check vs the native confidence: rows = checks, columns = subsets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e3df"
    color = {"residual": "#2a78d6", "confidence": "#eb6834"}  # categorical slots 1, 2
    subsets = [s for s in per_subset if per_subset[s]]
    fig, axes = plt.subplots(len(PLOT_CHECKS), len(subsets), figsize=(3.1 * len(subsets), 2.5 * len(PLOT_CHECKS)),
                             squeeze=False, constrained_layout=True)
    for i, name in enumerate(PLOT_CHECKS):
        for j, s in enumerate(subsets):
            ax = axes[i, j]
            m = per_subset[s].get(name, {})
            rc = m.get("risk_coverage")
            if not rc:
                ax.set_axis_off()
                continue
            cov = np.asarray(rc["coverage"], dtype=float) * 100
            mean_err = float(np.asarray(rc["oracle"], dtype=float)[-1])
            ax.axhline(mean_err, color=muted, lw=1, ls=":", zorder=1)
            ax.plot(cov, np.asarray(rc["oracle"], dtype=float), color=muted, lw=1.2, ls="--", zorder=2)
            for key in ("confidence", "residual"):
                ax.plot(cov, np.asarray(rc[key], dtype=float), color=color[key], lw=2, zorder=3,
                        solid_capstyle="round")
            ve = m["vs_error"]
            ax.text(0.97, 0.04, f"E-AURC\ncheck {ve['residual']['eaurc']:.3f}\nconf  {ve['confidence']['eaurc']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=muted, family="monospace",
                    multialignment="left", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 100)
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(grid)
            ax.tick_params(labelsize=7, colors=muted, length=0)
            if i == 0:
                ax.set_title(f"{s}", fontsize=9, color=ink, loc="left")
            if j == 0:
                ax.set_ylabel(f"{PLOT_LABEL[name]}\nrisk (mean rel. error)", fontsize=8, color=ink)
            if i == len(PLOT_CHECKS) - 1:
                ax.set_xlabel("coverage (% of queries kept)", fontsize=8, color=ink)
    handles = [plt.Line2D([], [], color=color["residual"], lw=2), plt.Line2D([], [], color=color["confidence"], lw=2),
               plt.Line2D([], [], color=muted, lw=1.2, ls="--"), plt.Line2D([], [], color=muted, lw=1, ls=":")]
    fig.legend(handles, ["check residual", "native confidence", "oracle (sorted by true error)", "keep all (mean error)"],
               loc="outside upper left", ncol=4, frameon=False, fontsize=8, labelcolor=ink)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/a3_detection")
    ap.add_argument("--from-json", action="store_true", help="only redraw the figure from an existing a3_detection.json")
    args = ap.parse_args()

    if args.from_json:
        write_png(json.loads((args.out / "a3_detection.json").read_text())["per_subset"], args.out / "risk_coverage.png")
        return
    subsets = args.subsets.split(",")
    start = time.time()
    pooled: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {s: {} for s in subsets}
    k_ratio: dict[str, list[float]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        rows, kr = clip_rows(d)
        k_ratio[entry["subset"]].append(kr)
        for name, r in rows.items():
            dst = pooled[entry["subset"]].setdefault(name, {})
            for k, v in r.items():
                dst.setdefault(k, []).append(v)
        print(f"[a3_det] {entry['file']}", flush=True)
    per_subset = {
        s: {name: score_check({k: np.concatenate(v) for k, v in cols.items()}) for name, cols in checks_.items()}
        for s, checks_ in pooled.items()
    }
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "checks": CHECKS, "per_subset": per_subset,
               "k_fit_ratio": k_ratio, "runtime_s": runtime}
    (args.out / "a3_detection.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, k_ratio, args.out / "a3_detection.md", sys.argv, runtime)
    write_png(jsonable(per_subset), args.out / "risk_coverage.png")
    print((args.out / "a3_detection.md").read_text())


if __name__ == "__main__":
    main()
