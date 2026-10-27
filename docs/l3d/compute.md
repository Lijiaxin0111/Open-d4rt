# L3D Part A: GPU jobs (G0 and query dumps)

Nothing on this page has been run yet. The code is tested only on CPU with a tiny
random model (`tests/l3d/test_engine.py`). Pick a backend (Modal or a CUDA box),
then run **fetch, then smoke, then gate0, then dump**, in that order.

## What runs

| Job | Modal (`scripts/l3d/modal_app.py`) | CUDA box | Does |
| --- | --- | --- | --- |
| fetch | `fetch_assets()` (CPU only, 2 cores, 4 GB, 2 h timeout) | `setup_cuda_box.sh` | 48CLIP `opend4rt.ckpt` + `model.yaml` from HF, plus the WorldTrack Drive folder via gdown. Safe to rerun: finished files are skipped. Prints file counts and sizes, and the npz keys of the first file in each subset (shows which subsets have `extrinsics_w2c`). |
| gate0 | `eval_gate0(subsets, limit_seqs)` (L40S, 32 GB RAM, 4 h) | `run_gate0.sh` | Runs the **unmodified** `eval_track3d_in_worldtrack.py --num-frames 64 --query-chunk-size 4096 --save-per-sequence` on the 48CLIP config. Gate 0 passes if each subset's APD(global) is within ±0.02 of the README value (Modal reports `gate0_pass` for each subset). |
| dump | `dump(subsets, limit_seqs, designs, ...)` (same GPU) | `run_dump.sh` | `scripts/l3d/dump_queries.py`: one encoder pass per 48-frame clip, then designs 0–4 decoded and design 5 decoded in a second pass from the design-4 `uv_2d`, into one `.npz` per clip (EXPERIMENTS.md section 3) plus `manifest.json` (args, git commit, ckpt, load info, timings, scale `s`, design-0 APD as a sanity check). Clips that already have a table are skipped, so a rerun picks up where it stopped. |
| smoke | `--task smoke` | the `LIMIT_SEQS=1` lines below | gate0 and dump on one `adt_mini` sequence (dump saves memory). |

Paths on the Modal volume `l3d-d4rt` (mounted at `/vol`): `checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/`,
`data/worldtrack_release/<subset>/*.npz`, `output/gate0_48clip/` (smoke run: `output/gate0_48clip_smoke/`), and
`output/l3d/dumps/<run>/` (default run name `OpenD4RT_48CLIP_9Mix_NoCropAUG_T48`; smoke run: `smoke_adt1`).
To change the GPU, RAM, or timeouts, edit the constants at the top of `modal_app.py` (`GPU_TYPE`,
`GPU_MEMORY_MB`, `GPU_TIMEOUT_S`, ...).

## Launch commands

Modal (needs `modal token new` once; `--detach` keeps the job running if the laptop disconnects):

```bash
.venv/bin/modal run scripts/l3d/modal_app.py --task fetch
.venv/bin/modal run scripts/l3d/modal_app.py --task smoke
.venv/bin/modal run --detach scripts/l3d/modal_app.py --task gate0
.venv/bin/modal run --detach scripts/l3d/modal_app.py --task dump
# options: --subsets po_mini,ds_mini --limit-seqs 5 --designs 0,1 --run-name NAME --clip-start 0 --save-memory --amp bf16
.venv/bin/modal volume get l3d-d4rt output/gate0_48clip/summary.txt .
.venv/bin/modal volume get l3d-d4rt output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48 output/l3d/dumps/
```

Modal uploads the local working tree (`src/`, `l3d/`, `scripts/`, root `*.py`, `checkpoints/*/model.yaml`), so
uncommitted code runs as-is. A CUDA box needs that code too: `rsync` the repo there (without `.venv`, `data`,
`output`), because nothing on this branch is pushed yet.

CUDA box (e.g. AWS g6e.xlarge: 1x L40S 48 GB, 4 vCPU, 32 GiB RAM; an Ubuntu 22.04 AMI with an NVIDIA
driver that supports CUDA 12.4, plus conda; about 100 GB of EBS):

```bash
bash scripts/l3d/setup_cuda_box.sh            # conda env "d4rt", 14 GB ckpt, about 6.6 GB WorldTrack
conda activate d4rt
python -m pytest tests/l3d -q                  # CPU tests
SUBSETS=adt_mini LIMIT_SEQS=1 OUTPUT_DIR=output/gate0_smoke bash scripts/l3d/run_gate0.sh
SUBSETS=adt_mini LIMIT_SEQS=1 RUN_NAME=smoke_adt1 SAVE_MEMORY=1 bash scripts/l3d/run_dump.sh
bash scripts/l3d/run_gate0.sh                  # -> output/gate0_48clip/summary.txt
bash scripts/l3d/run_dump.sh                   # -> output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48/
```

Environment-variable overrides: `EXP`, `DATA_ROOT`, `OUTPUT_DIR`, `SUBSETS`, `LIMIT_SEQS`, and for the dump also
`RUN_NAME`, `DESIGNS`, `CLIP_START`, `SAVE_MEMORY`, `AMP`, `CHUNK_SIZE`, `CKPT_MMAP`, `OVERWRITE`.

## WorldTrack release (Drive folder listing, sizes from HTTP HEAD, nothing downloaded)

`gdown.download_folder(url, skip_download=True)` lists 208 files:

| Folder | Files | Size | Per file |
| --- | ---: | ---: | --- |
| `adt_mini` | 50 | 1.68 GB | 25.8–36.1 MB |
| `po_mini` | 50 | 1.70 GB | 14.6–48.0 MB |
| `pstudio_mini` | 50 | 0.30 GB | 4.7–7.2 MB |
| `ds_mini` | 50 | 2.87 GB | 38.5–76.0 MB |
| `tum` (not used) | 8 | 0.08 GB | 8.8–10.2 MB |
| **total** | 208 | **6.63 GB** (6.55 GB for the 4 subsets) | |

Every subset folder lists exactly 50 files. Older gdown versions stopped at 50 files per folder; the
gdown 6.4.2 used here has no such cap, and 50 sequences per subset fits the README query totals. Still,
check that the G0 `queries` match the README (22187 / 53468 / 8720 / 52462).

## Storage

| Item | Size |
| --- | ---: |
| `opend4rt.ckpt` (48CLIP; 1.162 B params = 4.65 GB fp32, so the rest of the 13.95 GB is likely Adam state) | 13.95 GB |
| WorldTrack (4 subsets + tum) | 6.63 GB |
| Decoded tables: about 343k queries per 48x500-track clip = 23.8 MB raw, 10.3 MB `savez_compressed` (measured on a synthetic clip); 200 clips | about 2–3 GB |
| `--save-memory`: about 25 MB per clip (fp16 memory [1, 6145, 1280] = 15.7 MB, plus the 48x256x256x3 uint8 clip) | +5 GB for 200 clips |
| G0 outputs (JSON) | < 50 MB |

Loading the checkpoint briefly needs about 19 GB of RAM (4.65 GB model + 13.95 GB payload). It fits in the
32 GB Modal container and in g6e.xlarge's 32 GiB. `--ckpt-mmap` (dump only) lowers the peak.

## GPU time estimate (from the architecture; not measured)

FLOPs, counting a multiply-add as 2:

- **Encoder, one 48-frame clip**: ViT-g, 40 layers, width 1408, MLP 6144, 6144 video tokens + 1 aspect token.
  Linear layers: 50.5 MFLOP per token per layer, so 12.4 TFLOP. Global attention (20 layers, 6145 tokens): 4.25 TFLOP.
  Frame-local attention (20 layers, 24x256 tokens): 0.18 TFLOP. Total **about 17 TFLOP**.
- **Decoder, per query**: 8 cross-attention layers, width 1280, MLP 4480, attending to 6145 memory tokens:
  **about 0.49 GFLOP**. Every chunk of up to 4096 queries also recomputes the K/V projection of the memory:
  0.32 TFLOP per chunk.
- **L40S throughput**: upstream runs plain fp32 and does not enable TF32. Peak is 91.6 TFLOPS; assume 32–55
  TFLOPS effective.

**G0** (4 subsets x 50 sequences, 64 frames, `anchor_clip`): frames 0–47 share one encoder pass, and each of
frames 48–63 gets its own anchor clip (frame 0 + a 47-frame tail), so each sequence needs **17 encoder passes**.
That is 200 x 17 x 17 TFLOP = 57 PFLOP. Decoding is 2 x 64 x 136,837 frame-0 tracks = 17.5 M queries, or 8.6 PFLOP,
plus about 3.6 PFLOP of K/V overhead from the many small per-frame calls. Total about 69 PFLOP: **about 20–36 min
of GPU compute, 0.5–0.9 h wall** including the checkpoint load and CPU-side metrics (Sim3 RANSAC).

**Dump** (200 clips of 48 frames, all designs): a clip with N frame-0-visible tracks has 95N (design 0) +
73,728 (1) + 189,440 (2) + up to 24,576 (3) + up to 12,288 (4) + up to 61,440 (5, second pass) queries. That is
about 370–460k queries per clip, about 84 M in total: 41 PFLOP + 6.5 PFLOP K/V + 3.4 PFLOP encoder, about 51 PFLOP.
That comes to **about 15–27 min of GPU compute, 0.4–0.7 h wall** including compressed writes (about 1–2 s per clip). `--amp bf16` should cut the
decode time 3–5x, but the numbers would then differ from G0, so fp32 is the default.

## Cost estimate (prices are assumptions; check them before launching)

Prices assumed: Modal L40S $1.95/h GPU, plus about $0.35/h for the 32 GiB of RAM and a few CPU cores, so about
$2.3/h all-in (the Starter plan's monthly free credits may cover all of it). AWS g6e.xlarge in us-east-1:
$1.86/h on demand, spot about $0.6–1.0/h (varies by AZ), plus gp3 EBS at $0.08/GB-month (about $8/month for
100 GB). Modal volume storage pricing was not checked.

| Step | Wall time | Modal | AWS on-demand | AWS spot |
| --- | ---: | ---: | ---: | ---: |
| fetch / setup | 0.3–0.6 h | ~$0.10 (CPU only) | ~$0.6–1.1 | ~$0.3 |
| smoke (2 short GPU jobs, mostly checkpoint load) | ~0.3 h | ~$0.7 | ~$0.6 | ~$0.2 |
| gate0 | 0.5–0.9 h | ~$1.2–2.1 | ~$0.9–1.7 | ~$0.3–0.9 |
| dump | 0.4–0.6 h | ~$0.9–1.4 | ~$0.7–1.1 | ~$0.3–0.6 |
| **total** | ~1.5–2.4 h | **~$3–5** | **~$3–5** (+ EBS, idle time) | **~$1–2.5** |

On AWS the meter runs while the box idles, so stop the instance between steps. A spot interruption restarts G0
from the beginning, but the dump resumes (finished tables are skipped).

Known risks: Google Drive can rate-limit gdown from datacenter IPs ("too many users"). If that happens, rerun
fetch later; finished files are skipped. Modal L40S capacity can also queue.
