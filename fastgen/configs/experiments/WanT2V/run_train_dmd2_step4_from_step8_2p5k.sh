#!/usr/bin/env bash
set -euo pipefail

# Train progressive DMD2 stage: 8-step checkpoint -> 4-step student.
# Run from anywhere; this script cd's to the FastGen repo root.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PREV_EXP=${PREV_EXP:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8_2k}
PREV_CKPT=${PREV_CKPT:-$PREV_EXP/checkpoints/0002500}
CONFIG=${CONFIG:-fastgen/configs/experiments/WanT2V/config_dmd2_step4_from_step8_2p5k.py}
LOG_DIR=${LOG_DIR:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step4_from_step8_2p5k/logs}
mkdir -p "$LOG_DIR"

if [ ! -f "$PREV_EXP/config.yaml" ]; then
    echo "ERROR: previous-stage config not found: $PREV_EXP/config.yaml" >&2
    exit 1
fi

if [ ! -d "${PREV_CKPT}.net_model" ] || [ ! -f "${PREV_CKPT}.pth" ]; then
    echo "ERROR: previous-stage checkpoint prefix is incomplete: $PREV_CKPT" >&2
    echo "Expected ${PREV_CKPT}.net_model and ${PREV_CKPT}.pth" >&2
    exit 1
fi

PREV_STEPS=$(python3 - "$PREV_EXP/config.yaml" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
for line in text.splitlines():
    if line.strip().startswith("student_sample_steps:"):
        value = line.split(":", 1)[1].strip()
        print(value.replace("'", "").replace('"', ""))
        break
else:
    print("")
PY
)

if [ "$PREV_STEPS" != "8" ]; then
    echo "ERROR: previous-stage run is not an 8-step student." >&2
    echo "Found student_sample_steps=$PREV_STEPS in $PREV_EXP/config.yaml" >&2
    echo "Rerun the 8-step stage with config.model.student_sample_steps = 8 before launching this stage." >&2
    exit 1
fi

echo "Using previous 8-step checkpoint prefix: $PREV_CKPT"
echo "Using config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

LOG_FILE="$LOG_DIR/train_step4_from_step8_$(date +%Y%m%d_%H%M%S).log"
echo "Writing train log to $LOG_FILE"

torchrun --standalone --nproc_per_node=8 train.py     --config="$CONFIG"     2>&1 | tee "$LOG_FILE"
