# Paper vs. Code: D4RT implementation gap analysis

This document compares the **D4RT paper** (`docs/D4RT_paper.pdf`, "Efficiently
Reconstructing Dynamic Scenes One D4RT at a Time", Google DeepMind) against this
repository's **OpenD4RT** code.

> **Context.** OpenD4RT is an *unofficial, from-scratch re-implementation*. The
> original D4RT weights and code were never released, and the paper does not
> specify every architectural detail (the SRT-style decoder internals, exact
> attention/position-embedding scheme, and data pipeline are described only at a
> high level). Where the paper is silent, this repo makes its own engineering
> choices. "Consistent with the paper" below therefore means *consistent with
> what the paper actually states*, not bit-for-bit identical to the unreleased
> original.

Legend: ✅ implemented & faithful · ⚠️ implemented but a re-implementer's choice
where the paper is silent · ❌ missing / stubbed / diverges.

---

## ✅ Faithful to the paper

| Paper element | Code location |
| --- | --- |
| Encode video once → frozen *Global Scene Representation* `F`; decoder re-uses it for all queries | `src/model/d4rt.py:250-284` |
| Independent query interface `q=(u,v,t_src,t_tgt,t_cam) → P∈ℝ³` | `src/model/query_embedding.py`, `infer_track_3d.py` |
| ViT-g encoder: hidden 1408, 40 layers, patch `2×16×16`, interleaved local-framewise + global attention | `src/model/encoder.py:96-99,133-147`; `configs/model_effective.yaml:13-25` |
| VideoMAE2 pretrained init (key remap: `qkv→in_proj`, `q_bias`/`v_bias`→`in_proj_bias` w/ zero `k_bias`) | `src/model/d4rt.py:286-364` |
| Aspect-ratio token (resize-to-square + W/H token) | `src/model/d4rt.py:202-211`; `src/model/encoder.py:144` |
| 8-layer cross-attention decoder, **no query self-attention** ("queries do not interact") | `src/model/decoder.py` |
| Query embedding: Fourier(u,v) + learned discrete time embeddings + 9×9 local RGB patch | `src/model/query_embedding.py:10-39,59-71,80-118` |
| 13-dim output head: xyz(3)+uv(2)+vis(1)+disp(3)+normal(3)+conf(1) | `src/model/heads.py` |
| Loss: L1 xyz w/ `sign·log1p` + mean-depth normalize; L1 uv; BCE vis; L1 disp; cosine normal; `−log(c)` confidence penalty | `src/losses/d4rt_loss.py` |
| Training recipe: AdamW, wd 0.03, warmup→cosine, grad-clip L2=10, 20% hard queries near Sobel depth/motion boundaries, `P(t_tgt=t_cam)=0.4` | `src/engine/trainer.py:241-259,965-971`; `configs/train_effective.yaml:128-201` |
| 9-dataset training mixture (PointOdyssey, Dynamic Replica, Kubric, TartanAir, VKITTI2, ScanNet, BlendedMVS, CO3D, MVS-Synth) | `src/data/*_raw_dataset.py` (all 9 present) |
| Camera extrinsics via Umeyama Sim3; intrinsics from predicted points; long-video Sim3 stitching | `src/eval/tasks.py:11,72`; `infer_track_3d.py:176-216` |

---

## ⚠️ Re-implementer's own choices (paper is silent)

| Item | Code | Note |
| --- | --- | --- |
| Stock `nn.MultiheadAttention` + generic **1-D sinusoidal** position embedding | `src/model/encoder.py:122` | Not VideoMAE2's 3-D spatiotemporal pos-embed; the VideoMAE pos-embed is *not* loaded even when block weights are. |
| `_token_cap` adaptive-avg-pool to cap tokens at 6144 | `src/model/encoder.py:101-110` | Not in the paper; a memory guard. |
| Decoder width 1280 ≠ encoder 1408, bridged by `memory_proj` Linear | `src/model/d4rt.py:213` | Paper never specifies decoder width. |
| Anchor-clip / sliding-window stitching for videos longer than clip length | `infer_track_3d.py:89-155` | A plausible reading of appendix B; exact scheme is the author's. |

---

## ❌ Missing, stubbed, or divergent

| Item | Status | Evidence |
| --- | --- | --- |
| **Algorithm 1** — occupancy-grid efficient dense all-pixel tracking | **Now implemented in this fork** (was config-only stub) | `dense_track_occupancy_grid` in `infer_track_3d.py` + CLI `infer_dense_track.py`. Pure inference-time scheduler over the frozen model — no retraining. Originally only the string `occupancy_grid_tracking_all_pixels` at `configs/model_effective.yaml:74-76` existed. See "Why it was absent" below. |
| Full evaluation suite (TAPVid-3D cam/world tracking, Sintel/ScanNet/KITTI/Bonn depth, camera-pose ATE/RPE, point cloud, FPS/throughput, long-video KITTI) | **Not implemented** | Only WorldTrack sparse 3D-tracking exists (`eval_track3d_in_worldtrack.py`). README ToDo confirms full eval is "planned." |
| Waymo Open dataset (part of the paper's mixture) | **Dropped** | Only referenced as a camera convention in `docs/data_schema.md:161`; not in the training mixture. |
| SynthVerse (10Mix configs) | **Not released** | Checkpoint Zoo rows marked "Coming." |
| `λnormal` loss weight | **Divergent: 0.4 vs paper 0.5** | `configs/train_effective.yaml:161` |
| `reprojection_uv_from_xyz` Huber loss | **Invented extra, off by default** | `src/losses/d4rt_loss.py:161-213` |
| `lconf_ablation` confidence variant | **Speculative, off by default** | Code itself notes the paper "does not clearly specify a unique formula" (`src/losses/d4rt_loss.py:70-78`). |

---

## Why Algorithm 1 was absent despite being the paper's efficiency centerpiece

> **Update:** it is now implemented here as `dense_track_occupancy_grid`
> (`infer_track_3d.py`) with the `infer_dense_track.py` CLI. The notes below
> explain why the original release shipped without it.

Algorithm 1 is a **pure inference-time accelerator for the dense, all-pixel
tracking task** — orthogonal to correctness and to every metric this repo
reports:

1. **It changes speed, not output.** It maintains an occupancy grid, seeds new
   tracks only from un-visited pixels, and marks pixels a track passes through as
   visited. Naive full-grid querying yields the *same* trajectories; Alg. 1 only
   skips re-tracking already-covered pixels (paper frames it as a 5–15× adaptive
   speedup, not an accuracy feature).
2. **Training never needs it** — training uses sparse queries
   (`queries_per_clip: 4096`, `configs/train_effective.yaml:129`), not all pixels.
3. **The only released eval never needs it** — WorldTrack evaluates sparse
   first-frame-visible queries (`eval_track3d_in_worldtrack.py:503-521`), an
   already-small fixed query set.
4. **The dense demo uses naive querying** — a fixed `_grid_query_points` grid
   (`infer_track_3d.py:68-86`), no occupancy grid.

So the dense all-pixel path that would benefit from Alg. 1 simply isn't wired up
in the public release; the config entry marks intended behavior only.

---

## Practical implications

- **To run/evaluate:** download a checkpoint from HuggingFace
  (`Lijiaxin0111/OpenD4RT`, two real ~14 GB `opend4rt.ckpt` files) and use the
  WorldTrack eval / `infer_track_3d.py`. No training required. Checkpoints are
  git-ignored (`/checkpoints/**/*.ckpt`); only `model.yaml` ships in-repo.
- **Reported numbers are self-reported** for this re-implementation, not the
  official D4RT, and only on the WorldTrack subset.
- **To train/reproduce:** you additionally need the VideoMAE2 ViT-g init
  (`vit_g_hybrid_pt_1200e.pth` via `VIDEOMAE2_CKPT`) and all 9 raw datasets.
- Dense, high-throughput all-pixel 4D reconstruction (the Alg. 1 use case) is not
  available out of the box.
