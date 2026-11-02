#!/usr/bin/env bash
# One-time setup of a plain CUDA Linux box (e.g. AWS g6e.xlarge, 1x L40S 48 GB) for G0 and dumps:
# conda env from environment.yml, the 48CLIP checkpoint from HF (14 GB), WorldTrack from Google Drive (~6.6 GB).
# Needs ~30 GB free disk. Idempotent: finished steps are skipped on rerun.
#
#   bash scripts/l3d/setup_cuda_box.sh
#   conda activate d4rt && SUBSETS=adt_mini LIMIT_SEQS=1 bash scripts/l3d/run_gate0.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-d4rt}"
EXP_NAME="${EXP_NAME:-OpenD4RT_48CLIP_9Mix_NoCropAUG}"
HF_REPO="${HF_REPO:-Lijiaxin0111/OpenD4RT}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
WORLDTRACK_URL="${WORLDTRACK_URL:-https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV}"

# 1. Conda env (python 3.10, torch 2.6.0+cu124) + download tools.
if [[ "${SKIP_ENV:-0}" == "0" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Install Miniforge/Miniconda (or use a Deep Learning AMI) first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] conda env $ENV_NAME exists"
  else
    conda env create -n "$ENV_NAME" -f environment.yml
  fi
  conda activate "$ENV_NAME"
  pip install "huggingface_hub==2.3.0" "gdown==6.4.2" "scipy==1.15.3" pytest
  python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
fi

# 2. Checkpoint + model.yaml into checkpoints/<EXP_NAME>/.
CKPT="checkpoints/$EXP_NAME/opend4rt.ckpt"
if [[ "${SKIP_CKPT:-0}" == "0" ]]; then
  if [[ -f "$CKPT" && $(stat -c %s "$CKPT" 2>/dev/null || stat -f %z "$CKPT") -ge 13950000000 ]]; then
    echo "[setup] $CKPT present"
  else
    if command -v hf >/dev/null 2>&1; then HF_CLI=(hf download); else HF_CLI=(huggingface-cli download); fi
    "${HF_CLI[@]}" "$HF_REPO" \
      --repo-type model \
      --include "checkpoints/$EXP_NAME/opend4rt.ckpt" \
      --include "checkpoints/$EXP_NAME/model.yaml" \
      --local-dir .
  fi
  ls -la "checkpoints/$EXP_NAME"
fi

# 3. WorldTrack release into data/worldtrack_release/{adt_mini,po_mini,pstudio_mini,ds_mini}/*.npz (+ tum/).
if [[ "${SKIP_DATA:-0}" == "0" ]]; then
  mkdir -p "$DATA_ROOT"
  # --continue skips fully downloaded files, so a rerun only fetches what is missing.
  gdown --folder "$WORLDTRACK_URL" -O "$DATA_ROOT" --continue --retries 3
  for subset in adt_mini po_mini pstudio_mini ds_mini; do
    count=$(find "$DATA_ROOT/$subset" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    size=$(du -sh "$DATA_ROOT/$subset" 2>/dev/null | cut -f1)
    echo "[setup] $subset: $count files, $size (expected 50 files)"
  done
fi
echo "[setup] done. Next: conda activate $ENV_NAME && SUBSETS=adt_mini LIMIT_SEQS=1 bash scripts/l3d/run_gate0.sh"
