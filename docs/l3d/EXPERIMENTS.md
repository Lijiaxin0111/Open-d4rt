# L3D Part A: initial experiments (contract)

Shared contract for everyone (people and agents) writing Part A code. The project
plan is `docs/l3d/proposal_main.tex` (local only, not committed: the repo is
public). Meeting notes live in the team Google Doc.

Branch: `alicia/part-a-initial`. Do not commit or push without Alicia's OK.
Python env: `.venv/` at the repo root (CPU torch 2.6.0 with MPS, numpy 2.2.5,
scipy, pytest, modal, huggingface_hub, gdown). Run tests with
`.venv/bin/python -m pytest tests/l3d -q`.

## 1. What "initial experiments" means (this week)

| ID | What | Needs | Owner file(s) |
| --- | --- | --- | --- |
| G0 | Reproduce README WorldTrack numbers for `OpenD4RT_48CLIP_9Mix_NoCropAUG` (Gate 0: within ±2 APD per subset) | GPU, 14 GB ckpt, WorldTrack data | upstream `eval_track3d_in_worldtrack.py` (unchanged); result `results/l3d/gate0/` |
| Q | Decode arbitrary queries `(u, v, t_s, t_t, t_c)` against one cached encoder pass, not only whole tracks | GPU for real runs; CPU for tests | `l3d/engine.py`, `l3d/worldtrack_clip.py` |
| DUMP | Write decoded-query tables (schema in section 3) for the query designs in section 4 (design 5 in a second decode pass) | GPU | `scripts/l3d/dump_queries.py` |
| A2 | Sensitivity test of the five checks on a synthetic scene with injected errors (Table 1 of the proposal confirmed or corrected) | CPU only | `l3d/sensitivity.py`, `scripts/l3d/run_sensitivity_synthetic.py` |
| A1 | GT error decomposition: image-plane, depth offset/jitter, camera-frame (Sim(3) gauge + non-rigid) | CPU, on dumped tables | `l3d/decompose.py`, `scripts/l3d/run_a1_decompose.py` |
| A3 | T/S/C split (two-way ANOVA over source x camera frames), S share vs `abs(t_s - t_t)`; Spearman, AUROC, risk-coverage per check vs native confidence | CPU, on dumped tables | `l3d/tsc.py`, `l3d/detection.py`, `l3d/check_rows.py`, `scripts/l3d/run_a3_tsc.py`, `scripts/l3d/run_a3_detection.py` |
| A4 | Pose baselines: Umeyama, confidence-weighted Umeyama, ray center (adt_mini, po_mini, ds_mini have GT poses; pstudio_mini is a fixed camera) | CPU, on dumped tables | `l3d/checks.py`, `l3d/geometry.py`, `l3d/check_rows.py`, `scripts/l3d/run_a4_pose.py` |
| B | WorldTrack baselines: OpenD4RT vs ARROW vs TrackEverything on the same queries, under the Gate 0 rule and ARROW's in-image rule | GPU (each model in its own env); scoring on CPU with WorldTrack data | `scripts/l3d/run_opend4rt_worldtrack_preds.py`, `run_arrow_worldtrack_ourqueries.py`, `run_trackeverything_worldtrack.py`, `score_worldtrack_preds.py`, `run_baselines_table.py`; result `results/l3d/baselines/` |

## 2. Conventions

- **Clip.** A clip is `T <= 48` consecutive frames of one WorldTrack sequence
  (default: frames `[0, 48)`), resized to the model's 256x256, with the
  aspect-ratio token `W_native / H_native`. One encoder pass per clip; every
  query of that clip is decoded against the same cached memory, so `t_s`,
  `t_t`, `t_c` are clip-local indices in `[0, T)`. (Upstream eval on 64 frames
  uses `anchor_clip` re-encoding; that is only for G0.)
- **Pixels.** Normalized like upstream eval: `u = x_px / (W_native - 1)`,
  `v = y_px / (H_native - 1)`, in `[0, 1]`. The model's `uv_2d` head uses the
  same normalization. `K_gt` is in native pixels (`fx_fy_cx_cy` from the npz).
- **Cameras.** Right-handed, `+x` right, `+y` down, `+z` forward. WorldTrack
  `tracks_XYZ[t, n]` is track `n` at frame `t` in camera `t`'s own coordinates.
  `extrinsics_w2c[t]` is re-based so world = camera of clip frame 0 (as in
  `load_worldtrack_sequence`); if absent (Panoptic Studio, fixed camera) use
  identity. GT point of track `n` at time `t` in camera `c`:
  `X_c = T_w2c[c] @ inv(T_w2c[t]) @ X_t(n)`.
- **Model outputs.** `xyz_3d` is the point at time `t_t` in camera `t_c`'s
  coordinates, up to one unknown global scale. `visibility` and `confidence`
  are logits; use `sigmoid` (the loss uses `sigmoid(confidence)`).
- **Scale.** One global median-norm scale per clip,
  `s = median(|X_gt|) / median(|P_hat|)` over all finite self-frame
  (`t_c == t_t`) outputs with GT, exactly like
  `_compute_scale_factor_global` in `eval_track3d_in_worldtrack.py`, excluding
  design 5 (its source pixel is the model's, not GT; `l3d.decompose.table_scale`).
  Report `s`.
- **Depth-normalized residuals.** Any 3D residual used as a check is divided by
  the mean depth (z) of the points involved, so shrinking the scene cannot lower it.

## 3. Decoded-query table (`.npz`, one file per clip)

Written by `scripts/l3d/dump_queries.py`, read by every analysis module.
Path: `<results_root>/<run_name>/<subset>/<sequence>__c<clip_start>.npz`.
Arrays not listed as optional are required. `M` queries, `N` tracks, `T` frames.

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `subset`, `sequence` | str (0-d) | e.g. `po_mini`, `cnb_dlab_0215_3rd` |
| `clip_start`, `T` | int (0-d) | clip = native frames `[clip_start, clip_start + T)` |
| `hw_native` | `[2]` int | `(H, W)` of the native frames |
| `K_gt` | `[3, 3]` f64 | native-pixel intrinsics |
| `T_w2c_gt` | `[T, 4, 4]` f64 | world = clip frame 0 |
| `has_pose_gt` | bool (0-d) | False when the npz has no `extrinsics_w2c` |
| `gt_xyz_cam` | `[T, N, 3]` f32 | tracks in own-camera coords (NaN invalid) |
| `gt_uv` | `[T, N, 2]` f32 | normalized GT pixels |
| `gt_vis` | `[T, N]` bool | GT visibility |
| `is_static` | `[N]` bool | GT world displacement over the clip `< 0.01 * median depth` |
| `design` | `[M]` int8 | query design id (section 4) |
| `q_track` | `[M]` int32 | track index, or `-1` for a grid pixel with no GT track |
| `q_parent` | `[M]` int32 | design 5: row (in this table) of the design-4 query whose `p_uv` is this row's source pixel; `-1` for every other design |
| `q_u`, `q_v` | `[M]` f32 | source pixel (normalized) |
| `q_ts`, `q_tt`, `q_tc` | `[M]` int16 | source, target, camera frame |
| `p_xyz` | `[M, 3]` f32 | predicted point (model scale, camera `q_tc`) |
| `p_uv` | `[M, 2]` f32 | predicted 2D position at `q_tt` (normalized) |
| `p_vis_logit`, `p_conf_logit` | `[M]` f32 | raw logits |
| `g_xyz` | `[M, 3]` f32 | GT point at `q_tt` in camera `q_tc` (NaN if `q_track == -1`) |
| `g_uv` | `[M, 2]` f32 | GT pixel at `q_tt` (NaN if no track) |
| `g_vis` | `[M]` bool | GT visibility at `q_tt` |
| `model_ckpt`, `model_config` | str (0-d) | provenance |
| `memory_path` | str (0-d), optional | cached encoder memory (`.pt`, fp16) if saved |

## 4. Query designs

| id | Name | Queries | Used by |
| --- | --- | --- | --- |
| 0 | `track` | frame-0-visible tracks (all of them), `t_s = 0`, every `t_t`, `t_c in {t_t, 0}` | sanity vs upstream eval inside one clip; A1 |
| 1 | `cross` | up to 256 tracks; 8 target frames evenly spaced; for each `(n, t_t)`, 6 source frames where `n` is GT-visible (always include `t_t` if visible), GT source pixel; 6 camera frames always including `t_t` and 0; full crossed grid `S x C` | A3 T/S/C split; source-swap and camera-swap residuals; A1 camera-frame error |
| 2 | `grid` | 32x32 pixel grid of frame `t` (`t_s = t_t = t`), decoded in `t_c = t` and in `t_c = c` for `c in {0, t-4, t+4}` (clipped to `[0, T)`), for every `t` | camera swap, ray center, pose baselines (A4), intrinsics fit |
| 3 | `static` | GT-static tracks (up to 256), `t_s` = first visible frame, every `t_t`, `t_c in {0, T // 2}` | static rigidity |
| 4 | `selfmatch` | for each design-1 group `(n, t_t)`: anchor `t1` = `t_t` if it is a source, else the nearest source (earlier on ties); `t_s = t1` with GT pixel, `t_t = t_c = t2` for the group's sources, so `p_uv` gives the model's own `u2 = uv_hat(u1, t1, t2, t2)`; duplicate `(n, t1, t2)` rows dropped | parents of design 5 |
| 5 | `swap` | second decode pass: for each design-1 group `(n, t3)` with anchor `t1`, every source `t2 != t1` and every group camera `c`: `t_s = t2`, `t_t = t3`, `t_c = c`, source pixel = the parent design-4 `p_uv` of `(n, t1, t2)` clipped to `[0, 1]` (`q_parent`); rows whose parent `p_uv` is non-finite are dropped. `g_xyz` is track `n`'s GT, as if `u2` were right | source swap with model correspondences: `|P_5 - P_1(n, t1, t3, c)|` vs the GT-`u2` control `|P_1(n, t2, t3, c) - P_1(n, t1, t3, c)|`; `l3d.table.source_swap_rows` returns the row triples |

## 5. Results layout

- Small, committable summaries: `results/l3d/<experiment>/` (`.json`, `.md`, `.png`). Gate 1 combines them:
  `scripts/l3d/run_gate1.py` -> `results/l3d/gate1/`. Gate 3's self-correspondence error (design 4):
  `scripts/l3d/run_gate3_selfcorr.py` -> `results/l3d/gate3_selfcorr/`. Risk-coverage figure:
  `results/l3d/a3_detection/risk_coverage.png` (`run_a3_detection.py --from-json` redraws it).
- T/S/C figure: `results/l3d/a3_tsc/a3_tsc.png` (`run_a3_tsc.py --from-json`). GAUGE baseline (GT anchors, a
  reference): `l3d/gauge.py`, `scripts/l3d/run_gauge.py` -> `results/l3d/gauge/`. Bootstrap 95% CIs of the gate
  numbers over clips: `scripts/l3d/run_bootstrap.py` -> `results/l3d/bootstrap/` (run after the other results).
- Splits by GT visibility (at `t_t`), motion (`is_static`) and parallax (angle at the GT point between cameras
  `t_s` and `t_t`; bins 0-1, 1-5, >= 5 degrees): `l3d/splits.py`, `scripts/l3d/run_splits.py` -> `results/l3d/splits/`
  (tracking APD/EPE, self-frame depth share and T/S/C shares per bin).
- Report headers name the model from the dump manifest's `run_name` (`l3d.report.model_label`); every runner takes
  `--model-label` to override it (e.g. for another model's dump).
- WorldTrack baselines: each model saves `<subset>/<seq>.npz` (`pred_tracks_ref0`, `query_uv`) under
  `output/l3d/baselines/<model>/`; `score_worldtrack_preds.py` scores them with the upstream Gate 0 metric code
  (`--in-image` for ARROW's query rule, `--self-test` for a GT sanity check), and `run_baselines_table.py`
  -> `results/l3d/baselines/`. Save `query_uv` as float64: a few ds queries project to |u| ~ 1e6 px, where float32
  rounding fails the scorer's 1e-2 px match check.
- GT poses: `po_mini` extrinsics are off from orthonormal by up to ~1e-3, so every analysis projects rotations
  to SO(3) (`l3d.geometry.rigidify`, used by `relative_pose`).
- Large artifacts (decoded tables, cached memory, logs) stay off git:
  `output/l3d/...` locally, or the remote volume/bucket.
