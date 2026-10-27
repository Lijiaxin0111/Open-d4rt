#!/usr/bin/env bash
# Gate 0 on a plain CUDA box: the UNMODIFIED upstream WorldTrack eval for the 48CLIP checkpoint
# (64 frames, anchor-clip re-encoding). Compare summary.txt with the README table (within +-0.02 APD).
#
#   bash scripts/l3d/run_gate0.sh                                  # full G0, 4 subsets
#   SUBSETS=adt_mini LIMIT_SEQS=1 OUTPUT_DIR=output/gate0_smoke bash scripts/l3d/run_gate0.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EXP="${EXP:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gate0_48clip}"
SUBSETS="${SUBSETS:-adt_mini,po_mini,pstudio_mini,ds_mini}"
LIMIT_SEQS="${LIMIT_SEQS:-0}"
NUM_FRAMES="${NUM_FRAMES:-64}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

for path in "$EXP/model.yaml" "$EXP/opend4rt.ckpt" "$DATA_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing $path (run scripts/l3d/setup_cuda_box.sh first)." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
start_ts=$(date +%s)
"$PYTHON" eval_track3d_in_worldtrack.py \
  --model-config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --data-root "$DATA_ROOT" \
  --subsets "$SUBSETS" \
  --num-frames "$NUM_FRAMES" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --limit-seqs "$LIMIT_SEQS" \
  --save-per-sequence 2>&1 | tee "$OUTPUT_DIR/run_gate0.stdout.log"
echo "[gate0] wall time: $(( $(date +%s) - start_ts ))s; summary: $OUTPUT_DIR/summary.txt"
