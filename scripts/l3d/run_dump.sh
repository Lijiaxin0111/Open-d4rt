#!/usr/bin/env bash
# Decoded-query tables (docs/l3d/EXPERIMENTS.md sections 3-4) on a plain CUDA box.
#
#   bash scripts/l3d/run_dump.sh                                   # all 4 subsets, clip [0, 48)
#   SUBSETS=adt_mini LIMIT_SEQS=1 RUN_NAME=smoke_adt1 SAVE_MEMORY=1 bash scripts/l3d/run_dump.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EXP="${EXP:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
OUTPUT_DIR="${OUTPUT_DIR:-output/l3d/dumps}"
SUBSETS="${SUBSETS:-adt_mini,po_mini,pstudio_mini,ds_mini}"
LIMIT_SEQS="${LIMIT_SEQS:-0}"
RUN_NAME="${RUN_NAME:-$(basename "$EXP")_T48}"
DESIGNS="${DESIGNS:-0,1,2,3,4,5}"
CLIP_START="${CLIP_START:-0}"
CLIP_LEN="${CLIP_LEN:-48}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
AMP="${AMP:-none}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

for path in "$EXP/model.yaml" "$EXP/opend4rt.ckpt" "$DATA_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing $path (run scripts/l3d/setup_cuda_box.sh first)." >&2
    exit 1
  fi
done

ARGS=()
if [[ "${SAVE_MEMORY:-0}" != "0" ]]; then
  ARGS+=(--save-memory)
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${CKPT_MMAP:-0}" != "0" ]]; then
  ARGS+=(--ckpt-mmap)
fi

mkdir -p "$OUTPUT_DIR/$RUN_NAME"
start_ts=$(date +%s)
"$PYTHON" scripts/l3d/dump_queries.py \
  --model-config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --data-root "$DATA_ROOT" \
  --subsets "$SUBSETS" \
  --limit-seqs "$LIMIT_SEQS" \
  --clip-start "$CLIP_START" \
  --clip-len "$CLIP_LEN" \
  --designs "$DESIGNS" \
  --out-root "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --chunk-size "$CHUNK_SIZE" \
  --amp "$AMP" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$OUTPUT_DIR/$RUN_NAME/run_dump.stdout.log"
echo "[dump] wall time: $(( $(date +%s) - start_ts ))s; manifest: $OUTPUT_DIR/$RUN_NAME/manifest.json"
