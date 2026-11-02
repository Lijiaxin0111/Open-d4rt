"""Modal app for L3D Part A GPU jobs: asset fetch, Gate 0 (upstream WorldTrack eval), query dumps.

Nothing here runs until someone calls `modal run`; see docs/l3d/compute.md.

    modal run scripts/l3d/modal_app.py --task fetch
    modal run scripts/l3d/modal_app.py --task smoke
    modal run scripts/l3d/modal_app.py --task gate0
    modal run scripts/l3d/modal_app.py --task dump

Volume `l3d-d4rt` (mounted at /vol):
    /vol/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/{opend4rt.ckpt,model.yaml}
    /vol/data/worldtrack_release/{adt_mini,po_mini,pstudio_mini,ds_mini}/*.npz
    /vol/output/gate0_48clip/            upstream eval outputs (summary.json, per-sequence JSON)
    /vol/output/l3d/dumps/<run>/         decoded-query tables + manifest.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------- knobs (edit here)
APP_NAME = "l3d-d4rt"
VOLUME_NAME = "l3d-d4rt"
VOL = "/vol"
GPU_TYPE = "L40S"
GPU_MEMORY_MB = 32768
GPU_TIMEOUT_S = 4 * 60 * 60
FETCH_TIMEOUT_S = 2 * 60 * 60
FETCH_MEMORY_MB = 4096
FETCH_CPU = 2.0

EXP_NAME = "OpenD4RT_48CLIP_9Mix_NoCropAUG"
HF_REPO = "Lijiaxin0111/OpenD4RT"
WORLDTRACK_FOLDER_URL = "https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV"
SUBSETS_ALL = "adt_mini,po_mini,pstudio_mini,ds_mini"
# README WorldTrack numbers for OpenD4RT_48CLIP_9Mix_NoCropAUG (anchor_clip, 64 frames). Gate 0: |dAPD| <= 0.02.
README_APD_GLOBAL = {"adt_mini": 0.7220, "po_mini": 0.6799, "pstudio_mini": 0.7960, "ds_mini": 0.7248}
GATE0_APD_TOL = 0.02

REPO_REMOTE = "/root/l3d-d4rt"
CKPT_DIR = f"{VOL}/checkpoints/{EXP_NAME}"
CKPT_PATH = f"{CKPT_DIR}/opend4rt.ckpt"
DATA_ROOT = f"{VOL}/data/worldtrack_release"
GATE0_OUT = f"{VOL}/output"
DUMP_OUT = f"{VOL}/output/l3d/dumps"

LOCAL_REPO = Path(__file__).resolve().parents[2]
# Only src/, l3d/, scripts/, root *.py and checkpoints/*/model.yaml are added, so the repo-root
# .venv, .git, output, tmp, data, results never reach the image. Patterns are relative to each
# added dir (a bare "data" here would wrongly drop src/data).
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.ckpt", "**/*.pth", "**/.DS_Store"]


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("libglib2.0-0", "git")
        # requirements.txt carries --extra-index-url for the torch 2.6.0+cu124 wheels.
        .pip_install_from_requirements(str(LOCAL_REPO / "requirements.txt"))
        .pip_install("huggingface_hub==2.3.0", "gdown==6.4.2", "scipy==1.15.3")
        .env({"PYTHONPATH": REPO_REMOTE, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    )
    for sub in ("src", "l3d", "scripts"):
        image = image.add_local_dir(LOCAL_REPO / sub, f"{REPO_REMOTE}/{sub}", ignore=_IGNORE)
    for path in sorted(LOCAL_REPO.glob("*.py")):
        image = image.add_local_file(path, f"{REPO_REMOTE}/{path.name}")
    for path in sorted(LOCAL_REPO.glob("checkpoints/*/model.yaml")):
        image = image.add_local_file(path, f"{REPO_REMOTE}/{path.relative_to(LOCAL_REPO).as_posix()}")
    return image


app = modal.App(APP_NAME)
image = _build_image()
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _model_yaml() -> str:
    # Prefer the HF copy fetched next to the checkpoint; the repo copy is identical in content.
    fetched = Path(CKPT_DIR) / "model.yaml"
    return str(fetched if fetched.exists() else Path(REPO_REMOTE) / "checkpoints" / EXP_NAME / "model.yaml")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_REMOTE, check=True, env={**os.environ, **(env or {})})


def _require_assets() -> None:
    volume.reload()
    missing = [p for p in (CKPT_PATH, DATA_ROOT) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing on volume {VOLUME_NAME}: {missing}. Run `--task fetch` first.")


def _dir_report(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        files = sorted(sub.glob("*.npz"))
        out[sub.name] = {"files": len(files), "gb": round(sum(f.stat().st_size for f in files) / 1e9, 3)}
    return out


# ---------------------------------------------------------------------- functions
@app.function(image=image, volumes={VOL: volume}, cpu=FETCH_CPU, memory=FETCH_MEMORY_MB, timeout=FETCH_TIMEOUT_S)
def fetch_assets() -> dict[str, Any]:
    """Download the 48CLIP checkpoint (HF) and the WorldTrack release (Google Drive) into the volume. Idempotent."""
    import gdown
    from huggingface_hub import hf_hub_download

    report: dict[str, Any] = {}
    start = time.time()
    for name in ("model.yaml", "opend4rt.ckpt"):
        # local_dir downloads skip files that are already complete (metadata under /vol/.cache/huggingface).
        path = hf_hub_download(repo_id=HF_REPO, repo_type="model", filename=f"checkpoints/{EXP_NAME}/{name}", local_dir=VOL)
        size = Path(path).stat().st_size
        report[name] = {"path": path, "gb": round(size / 1e9, 3)}
        print(f"[fetch] {path}: {size / 1e9:.2f} GB", flush=True)
    volume.commit()

    data_root = Path(DATA_ROOT)
    data_root.mkdir(parents=True, exist_ok=True)
    # resume=True skips fully downloaded files and resumes partial ones; the folder also holds `tum/` (~80 MB).
    gdown.download_folder(url=WORLDTRACK_FOLDER_URL, output=str(data_root), quiet=False, resume=True, retries=3)
    volume.commit()
    report["worldtrack"] = _dir_report(data_root)
    for subset in SUBSETS_ALL.split(","):
        info = report["worldtrack"].get(subset, {"files": 0, "gb": 0.0})
        files = sorted((data_root / subset).glob("*.npz"))
        if files:
            # Which subsets carry pose GT (`extrinsics_w2c`), and array shapes of the first sequence.
            import numpy as np

            with np.load(files[0], allow_pickle=True) as pack:
                info["first_file_keys"] = {k: list(pack[k].shape) for k in pack.files}
        print(f"[fetch] {subset}: {info['files']} files, {info['gb']:.2f} GB, keys={info.get('first_file_keys')}", flush=True)
    report["seconds"] = round(time.time() - start, 1)
    return report


@app.function(image=image, volumes={VOL: volume}, gpu=GPU_TYPE, memory=GPU_MEMORY_MB, timeout=GPU_TIMEOUT_S)
def eval_gate0(subsets: str = SUBSETS_ALL, limit_seqs: int = 0, output_name: str = "gate0_48clip") -> dict[str, Any]:
    """Gate 0: the UNMODIFIED upstream `eval_track3d_in_worldtrack.py` on the 48CLIP checkpoint, 64 frames."""
    _require_assets()
    output_dir = f"{GATE0_OUT}/{output_name}"
    cmd = [
        sys.executable,
        "eval_track3d_in_worldtrack.py",
        "--model-config", _model_yaml(),
        "--ckpt-path", CKPT_PATH,
        "--data-root", DATA_ROOT,
        "--subsets", subsets,
        "--num-frames", "64",
        "--query-chunk-size", "4096",
        "--output-dir", output_dir,
        "--device", "cuda",
        "--save-per-sequence",
    ]
    if int(limit_seqs) > 0:
        cmd += ["--limit-seqs", str(int(limit_seqs))]
    start = time.time()
    try:
        _run(cmd)
    finally:
        volume.commit()
    summary = json.loads((Path(output_dir) / "summary.json").read_text(encoding="utf-8"))
    # Drop per-sequence lists from the return value (they stay in summary.json on the volume).
    compact = {"inputs": summary.get("inputs", {}), "output_dir": output_dir, "seconds": round(time.time() - start, 1), "subsets": {}}
    for subset, payload in summary.get("subsets", {}).items():
        stats = {k: v for k, v in payload.items() if k != "sequences"}
        apd = float(stats.get("avg_pts_global", float("nan")))
        target = README_APD_GLOBAL.get(subset)
        stats["readme_apd_global"] = target
        stats["gate0_pass"] = bool(target is not None and abs(apd - target) <= GATE0_APD_TOL) if int(limit_seqs) <= 0 else None
        compact["subsets"][subset] = stats
    print(json.dumps(compact, indent=2), flush=True)
    return compact


@app.function(image=image, volumes={VOL: volume}, gpu=GPU_TYPE, memory=GPU_MEMORY_MB, timeout=GPU_TIMEOUT_S)
def dump(
    subsets: str = SUBSETS_ALL,
    limit_seqs: int = 0,
    designs: str = "0,1,2,3,4,5",
    run_name: str = "",
    clip_start: str = "0",
    save_memory: bool = False,
    amp: str = "none",
    git_commit: str = "",
    git_dirty: str = "",
) -> dict[str, Any]:
    """Decoded-query tables via `scripts/l3d/dump_queries.py` -> /vol/output/l3d/dumps/<run>."""
    _require_assets()
    run_name = run_name or f"{EXP_NAME}_T48"
    cmd = [
        sys.executable,
        "scripts/l3d/dump_queries.py",
        "--model-config", _model_yaml(),
        "--ckpt-path", CKPT_PATH,
        "--data-root", DATA_ROOT,
        "--subsets", subsets,
        "--limit-seqs", str(int(limit_seqs)),
        "--clip-start", str(clip_start),
        "--clip-len", "48",
        "--designs", designs,
        "--out-root", DUMP_OUT,
        "--run-name", run_name,
        "--device", "cuda",
        "--seed", "0",
        "--amp", amp,
    ]
    if save_memory:
        cmd.append("--save-memory")
    start = time.time()
    try:
        _run(cmd, env={"L3D_GIT_COMMIT": git_commit, "L3D_GIT_DIRTY": git_dirty})
    finally:
        volume.commit()
    manifest = json.loads((Path(DUMP_OUT) / run_name / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest.get("clips", [])
    keys = ("subset", "sequence", "clip_start", "status", "num_queries", "encode_s", "decode_s", "file_bytes", "d0_apd_global")
    out = {
        "run_dir": f"{DUMP_OUT}/{run_name}",
        "seconds": round(time.time() - start, 1),
        "load_info": manifest.get("load_info", {}),
        "num_clips": len(clips),
        "clips": [{k: c.get(k) for k in keys} for c in clips],
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


# ---------------------------------------------------------------------- local entrypoint
def _local_git() -> tuple[str, str]:
    try:
        commit = subprocess.run(["git", "-C", str(LOCAL_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(LOCAL_REPO), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return commit, str(bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


@app.local_entrypoint()
def main(
    task: str = "smoke",
    subsets: str = SUBSETS_ALL,
    limit_seqs: int = 0,
    designs: str = "0,1,2,3,4,5",
    run_name: str = "",
    clip_start: str = "0",
    save_memory: bool = False,
    amp: str = "none",
) -> None:
    commit, dirty = _local_git()
    if task == "fetch":
        print(json.dumps(fetch_assets.remote(), indent=2))
    elif task == "smoke":
        print(json.dumps(eval_gate0.remote(subsets="adt_mini", limit_seqs=1, output_name="gate0_48clip_smoke"), indent=2))
        result = dump.remote(
            subsets="adt_mini",
            limit_seqs=1,
            designs=designs,
            run_name=run_name or "smoke_adt1",
            clip_start=clip_start,
            save_memory=True,
            amp=amp,
            git_commit=commit,
            git_dirty=dirty,
        )
        print(json.dumps(result, indent=2))
    elif task == "gate0":
        print(json.dumps(eval_gate0.remote(subsets=subsets, limit_seqs=limit_seqs), indent=2))
    elif task == "dump":
        result = dump.remote(
            subsets=subsets,
            limit_seqs=limit_seqs,
            designs=designs,
            run_name=run_name,
            clip_start=clip_start,
            save_memory=save_memory,
            amp=amp,
            git_commit=commit,
            git_dirty=dirty,
        )
        print(json.dumps(result, indent=2))
    else:
        raise ValueError(f"Unknown --task {task!r}; expected fetch|smoke|gate0|dump")
