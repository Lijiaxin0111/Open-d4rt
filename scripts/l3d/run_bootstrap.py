#!/usr/bin/env python
"""Bootstrap 95% CIs of the gate numbers over clips; writes bootstrap.{json,md}.

Clips (one per sequence) are resampled with replacement within each subset and every
statistic is recomputed the way its gate pools it: ratios of per-clip sums for shares,
pooled medians / Spearman rho for row statistics (rows subsampled to ``--rows`` per clip
and check), medians over pairs for the A4 ratios, and the GAUGE EPE gain as a ratio of
clip means. The point estimates are the full-data values of the gate tables; a gate
verdict is "robust" when the whole CI is on one side of its threshold.

Example:
    .venv/bin/python scripts/l3d/run_bootstrap.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import check_rows as cr  # noqa: E402
from l3d import detection as det  # noqa: E402
from l3d import tsc  # noqa: E402
from l3d.decompose import table_scale  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable  # noqa: E402
from l3d.table import select  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
RES = REPO / "results/l3d"
CHECKS = ("source_swap", "camera_swap", "ray_center")


def _sub(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else rng.choice(n, size=k, replace=False)


def clip_ingredients(d: dict[str, Any], rng: np.random.Generator, k_rows: int) -> dict[str, Any]:
    """Everything one clip contributes to the resampled statistics."""
    s = table_scale(d)
    err = cr.row_errors(d, s)
    K = cr.fit_K(d, err)
    ss = cr.source_swap_detection(d, err)["model_u2"]
    cs = cr.camera_swap_detection(d, err)
    rc = cr.ray_center_detection(d, err, K)
    rows = {}
    for name, r, res_key in (("source_swap", ss, "residual"), ("camera_swap", cs, "residual_se3"),
                             ("ray_center", rc, "residual")):
        ok = np.isfinite(r[res_key]) & np.isfinite(r["target"]) & np.isfinite(r["error"])
        idx = np.flatnonzero(ok)[_sub(rng, int(ok.sum()), k_rows)]
        rows[name] = {"residual": r[res_key][idx], "target": r["target"][idx], "error": r["error"][idx]}
    sums = {}
    for comp in ("log_depth", "xyz"):
        tf = tsc.tensor_from_table(d, component=comp, scale=s)
        sp = tsc.two_way_split(tf["E"], tf["mask"])
        sums[comp] = {k: float(np.nansum(sp[k])) for k in ("T", "S", "C", "SC", "total")}
    pb = cr.pose_baselines(d, err)
    sm = select(d, design="selfmatch")
    keep = sm["g_vis"] & (sm["q_ts"] != sm["q_tt"]) & np.isfinite(sm["p_uv"]).all(-1) & np.isfinite(sm["g_uv"]).all(-1)
    e256 = np.linalg.norm((sm["p_uv"][keep] - sm["g_uv"][keep]).astype(np.float64) * 255.0, axis=-1)
    return {
        "rows": rows, "sums": sums,
        "pairs": {"cs": pb["pairs"]["cs_residual"], "rc": pb["pairs"]["rc_spread"],
                  "pose": pb["methods"]["umeyama"]["trans_rel"]},
        "selfcorr": e256[_sub(rng, e256.size, k_rows)],
    }


def a1_ingredients(a1: dict[str, Any], subset: str) -> list[dict[str, float]]:
    out = []
    for c in a1["clips"][subset]:
        if "error" in c or not c["self"]["n"]:
            out.append({"w_self": 0.0, "w_perp": 0.0, "w_depth": 0.0, "w_off": 0.0, "tot2": 0.0, "nr2": 0.0})
            continue
        w_self = c["self"]["n"] * c["self"]["sq_mean"]
        w_depth = c["self"]["n"] * c["depth_ms_total"]
        out.append({"w_self": w_self, "w_perp": w_self * (1 - c["self"]["radial_share"]), "w_depth": w_depth,
                    "w_off": w_depth * c["depth_shares"]["offset"], "tot2": c["camera_tot2"], "nr2": c["camera_nr2"]})
    return out


def statistics(clips: list[dict[str, Any]], a1c: list[dict[str, float]], gauge_c: list[dict[str, Any]] | None
               ) -> dict[str, Callable[[np.ndarray], float]]:
    """Statistic name -> function of a resampled clip index array."""
    cat = lambda idx, f: np.concatenate([f(clips[i]) for i in idx])  # noqa: E731
    ssum = lambda idx, f: float(sum(f(i) for i in idx))  # noqa: E731
    stats: dict[str, Callable[[np.ndarray], float]] = {}
    for c in CHECKS:
        stats[f"{c} (ii) rho"] = lambda idx, c=c: det.spearman(cat(idx, lambda x: x["rows"][c]["residual"]),
                                                               cat(idx, lambda x: x["rows"][c]["target"]))
    stats["source_swap (iii) S+SC share of depth"] = lambda idx: (
        ssum(idx, lambda i: clips[i]["sums"]["log_depth"]["S"] + clips[i]["sums"]["log_depth"]["SC"])
        / ssum(idx, lambda i: clips[i]["sums"]["log_depth"]["total"]))
    stats["camera_swap (iii) non-rigid share"] = lambda idx: ssum(idx, lambda i: a1c[i]["nr2"]) / ssum(idx, lambda i: a1c[i]["tot2"])
    stats["ray_center (iii) image-plane share"] = lambda idx: ssum(idx, lambda i: a1c[i]["w_perp"]) / ssum(idx, lambda i: a1c[i]["w_self"])
    stats["source_swap (iv) median ratio"] = lambda idx: float(
        np.median(cat(idx, lambda x: x["rows"]["source_swap"]["residual"])) / np.median(cat(idx, lambda x: x["rows"]["source_swap"]["error"])))
    pose = lambda idx: float(np.median(cat(idx, lambda x: x["pairs"]["pose"])))  # noqa: E731
    stats["camera_swap (iv) median ratio"] = lambda idx: float(np.median(cat(idx, lambda x: x["pairs"]["cs"]))) / pose(idx)
    stats["ray_center (iv) median ratio"] = lambda idx: float(np.median(cat(idx, lambda x: x["pairs"]["rc"]))) / pose(idx)
    stats["Gate 3 median px"] = lambda idx: float(np.median(cat(idx, lambda x: x["selfcorr"])))
    stats["RQ1 disagreement (1 - T), 3D"] = lambda idx: 1.0 - (
        ssum(idx, lambda i: clips[i]["sums"]["xyz"]["T"]) / ssum(idx, lambda i: clips[i]["sums"]["xyz"]["total"]))
    stats["RQ1 depth offset share"] = lambda idx: ssum(idx, lambda i: a1c[i]["w_off"]) / ssum(idx, lambda i: a1c[i]["w_depth"])
    if gauge_c is not None:
        def gain(idx: np.ndarray, part: str) -> float:
            v = [gauge_c[i] for i in idx if gauge_c[i] and gauge_c[i]["0.05"]["base"][part]["n"]]
            b = np.mean([x["0.05"]["base"][part]["epe"] for x in v])
            return float(100 * (b - np.mean([x["0.05"]["gauge"][part]["epe"] for x in v])) / b)
        stats["GAUGE 5% EPE gain %, all"] = lambda idx: gain(idx, "all")
        stats["GAUGE 5% EPE gain %, dyn"] = lambda idx: gain(idx, "dyn")
    return stats


# statistic -> (threshold, "ge" pass if >= threshold, "le" pass if <= threshold)
THRESH = {**{f"{c} (ii) rho": (0.3, "ge") for c in CHECKS},
          "source_swap (iii) S+SC share of depth": (0.2, "ge"), "camera_swap (iii) non-rigid share": (0.2, "ge"),
          "ray_center (iii) image-plane share": (0.2, "ge"), **{f"{c} (iv) median ratio": (0.2, "ge") for c in CHECKS},
          "Gate 3 median px": (2.0, "le"), "RQ1 disagreement (1 - T), 3D": (0.2, "ge"), "RQ1 depth offset share": (0.5, "ge")}


def point_estimates(subset: str) -> dict[str, float]:
    """Full-data values from the gate tables (bootstrap uses row subsamples for row statistics)."""
    load = lambda p: json.loads((RES / p).read_text())  # noqa: E731
    g1 = load("gate1/gate1.json")
    out = {}
    for c in CHECKS:
        v = g1[c]["subsets"][subset]
        out[f"{c} (ii) rho"] = v["rho"]
        out[f"{c} (iv) median ratio"] = v["ratio"]
    out["source_swap (iii) S+SC share of depth"] = g1["source_swap"]["subsets"][subset]["visible"]
    out["camera_swap (iii) non-rigid share"] = g1["camera_swap"]["subsets"][subset]["visible"]
    out["ray_center (iii) image-plane share"] = g1["ray_center"]["subsets"][subset]["visible"]
    out["RQ1 disagreement (1 - T), 3D"] = g1["rq1"][subset]["disagreement_xyz"]
    out["RQ1 depth offset share"] = g1["rq1"][subset]["depth_offset_share"]
    out["Gate 3 median px"] = load("gate3_selfcorr/selfcorr.json")["per_subset"][subset]["gt_visible"]["median"]
    return out


def verdict(name: str, lo: float, hi: float) -> str:
    if name not in THRESH or not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    th, kind = THRESH[name]
    ok_lo, ok_hi = (lo >= th, hi >= th) if kind == "ge" else (lo <= th, hi <= th)
    if ok_lo and ok_hi:
        return "robust pass"
    if not ok_lo and not ok_hi:
        return "robust fail"
    return "**uncertain**"


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, argv: list[str], runtime_s: float, b: int) -> None:
    lines = [
        "# Bootstrap 95% CIs of the gate numbers (OpenD4RT 48CLIP, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s, {b} resamples of each subset's clips, one per sequence). Point = full-data value of the",
        "gate tables (bootstrap row statistics use a per-clip row subsample). Verdict: the whole CI on one side of the",
        "threshold (pass = `>=` threshold, except Gate 3: `<=` 2 px). RQ1 rows: >= 20% disagreement, >= 50% offset.",
        "",
        "| Subset | Statistic | Threshold | Point | 95% CI | Verdict |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for s, r in per_subset.items():
        for name, v in r.items():
            th = THRESH.get(name)
            th_txt = "" if th is None else f"{'>=' if th[1] == 'ge' else '<='} {th[0]:g}"
            lines.append(f"| `{s}` | {name} | {th_txt} | {fmt(v['point'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | "
                         f"{verdict(name, v['lo'], v['hi'])} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--b", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--rows", type=int, default=1000, help="row subsample per clip and check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RES / "bootstrap")
    args = ap.parse_args()

    subsets = args.subsets.split(",")
    start = time.time()
    a1 = json.loads((RES / "a1_decompose/a1.json").read_text())
    gauge_path = RES / "gauge/gauge.json"
    gauge_clips = json.loads(gauge_path.read_text())["clips"] if gauge_path.exists() else None
    clips: dict[str, list[dict[str, Any]]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        rng = np.random.default_rng(args.seed + zlib.crc32(entry["file"].encode()))
        clips[entry["subset"]].append(clip_ingredients(d, rng, args.rows))
        print(f"[boot] {entry['file']}", flush=True)
    per_subset: dict[str, dict[str, Any]] = {}
    for s in subsets:
        n = len(clips[s])
        a1c = a1_ingredients(a1, s)[:n]
        gc = gauge_clips[s][:n] if gauge_clips is not None else None
        stats = statistics(clips[s], a1c, gc)
        points = point_estimates(s)
        rng = np.random.default_rng(args.seed)
        draws = [rng.integers(0, n, size=n) for _ in range(args.b)]
        per_subset[s] = {}
        for name, f in stats.items():
            vals = np.array([f(idx) for idx in draws])
            point = points.get(name, f(np.arange(n)))
            per_subset[s][name] = {"point": point, "lo": float(np.nanpercentile(vals, 2.5)),
                                   "hi": float(np.nanpercentile(vals, 97.5)), "boot_mean": float(np.nanmean(vals))}
            print(f"[boot] {s} {name}: {point:.4g} [{per_subset[s][name]['lo']:.4g}, {per_subset[s][name]['hi']:.4g}]",
                  flush=True)
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "thresholds": THRESH, "per_subset": per_subset,
               "runtime_s": runtime}
    (args.out / "bootstrap.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, args.out / "bootstrap.md", sys.argv, runtime, args.b)
    print((args.out / "bootstrap.md").read_text())


if __name__ == "__main__":
    main()
