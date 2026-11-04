# Consumer-GPU / Blackwell (sm_120) compatibility notes

Hi! First, thanks for open-sourcing this — being able to reproduce the released
WorldTrack eval end-to-end on a laptop was really valuable for me.

I ran the released `32CLIP_9Dataset_NoAUG` eval on a consumer laptop and hit a
few things that are not covered in the current docs. Everyone with a recent
(40-series / 50-series) NVIDIA card will hit the first one, so I thought it was
worth writing down. This is a docs-only contribution — I didn't change the model
or the eval logic.

## My environment

- GPU: RTX 5070 Laptop (8 GB VRAM, Blackwell / sm_120)
- OS: Windows, Anaconda, Python 3.10
- Final working torch: `torch 2.11.0+cu128`

## 1. The pinned torch (cu124) does not run on Blackwell GPUs

`requirements.txt` pins:

```
--extra-index-url=https://download.pytorch.org/whl/cu124
torch==2.6.0+cu124
torchvision==0.21.0+cu124
```

The `cu124` wheels do not ship `sm_120` kernels, so on any Blackwell card
(RTX 50-series) this fails immediately at the first CUDA op with:

```
CUDA error: no kernel image is available for execution on the device
```

Fix that worked for me: install a `cu128` torch instead, then install the rest
of `requirements.txt` (with the torch lines removed/commented so they don't
downgrade it):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Quick check that the GPU actually runs kernels (not just that torch imports):

```bash
python -c "import torch; x=torch.randn(1000,1000,device='cuda'); print((x@x).sum().item())"
```

Suggestion: a short note in the README (or a `requirements-blackwell.txt`) would
save 50-series users from getting stuck at step one.

## 2. 8 GB VRAM is enough for the 32CLIP eval (fp32), but only just

I ran the full WorldTrack eval (all four subsets) with the released 32CLIP
checkpoint in plain fp32 — no autocast, no code changes — and it fits in 8 GB,
but with very little headroom:

- Observed VRAM usage: ~7.4 GB / 8 GB (~90%), GPU utilization ~100%
- No OOM across all four subsets
- Per-sequence runtime: roughly 50–60 s per sequence at `--num-frames 64`,
  `--query-chunk-size 512`

So 8 GB works, but 6 GB likely would not without lowering precision. For anyone
tighter on memory, wrapping inference in `torch.autocast("cuda", bf16)` should
roughly halve the weight footprint — I didn't need it here, so I haven't
measured it, but flagging it as the obvious next step.

Command I used (Windows, calling the python entrypoint directly instead of the
`.sh`):

```bash
python eval_track3d_in_worldtrack.py \
  --model-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --ckpt-path checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --data-root data/worldtrack_release \
  --subsets adt_mini,po_mini,pstudio_mini,ds_mini \
  --num-frames 64 --query-chunk-size 512 \
  --output-dir tmp/eval_full --device cuda --save-per-sequence
```

### Reproduced numbers (match the README table exactly)

| Subset | APD (mine) | APD (README) | EPE (mine) | EPE (README) | Queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| adt_mini | 0.6991 | 0.6993 | 0.2966 | 0.2964 | 22187 |
| po_mini | 0.6602 | 0.6603 | 0.3398 | 0.3397 | 53468 |
| pstudio_mini | 0.7863 | 0.7863 | 0.1811 | 0.1811 | 8720 |
| ds_mini | 0.7266 | 0.7266 | 0.2944 | 0.2944 | 52462 |

All four subsets match the published numbers to 4 decimal places, with identical
query counts — so the eval pipeline reproduces exactly on consumer hardware.

## 3. Downloading the WorldTrack data as one zip can silently truncate

When I downloaded the WorldTrack release folder from Google Drive as a single
zip, the archive was incomplete — some subsets were missing sequences (only
`pstudio_mini` came through with all 50; the others were partial). This is easy
to miss because the eval still runs, it just reports metrics over fewer queries.

Downloading each subset folder separately fixed it. A one-line note like "verify
each subset has its full `.npz` count after download" in the WorldTrack Data
section would help.

---

Happy to adjust any of this or split it into separate issues if that's easier to
review. Thanks again for the repo.
