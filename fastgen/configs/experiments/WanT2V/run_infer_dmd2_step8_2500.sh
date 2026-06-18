#!/usr/bin/env bash
set -euo pipefail

# Reusable Wan2.1-T2V DMD2 8-step student inference.
# Defaults evaluate checkpoint 0002500 on the shared 10-prompt OpenVid eval file.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXP_ROOT=${EXP_ROOT:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8}
CKPT_STEP=${CKPT_STEP:-0002500}
CKPT_PATH=${CKPT_PATH:-$EXP_ROOT/checkpoints/$CKPT_STEP}
CONFIG=${CONFIG:-fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py}
PROMPT_FILE=${PROMPT_FILE:-scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt}
NEG_PROMPT_FILE=${NEG_PROMPT_FILE:-scripts/inference/prompts/negative_prompt.txt}
LOG_DIR=${LOG_DIR:-$EXP_ROOT/logs}
VIDEO_SAVE_DIR=${VIDEO_SAVE_DIR:-$EXP_ROOT/inference/eval_10promprs/${CKPT_STEP}_student_8step}
SAVE_HIGH_QUALITY=${SAVE_HIGH_QUALITY:-True}
FPS=${FPS:-16}
SEED=${SEED:-0}
FORCE=${FORCE:-0}

PROMPT_STEM=$(basename "$PROMPT_FILE" .txt)
FINAL_VIDEO_DIR="$VIDEO_SAVE_DIR/$PROMPT_STEM"
HQ_VIDEO_DIR="${FINAL_VIDEO_DIR}_hq"

mkdir -p "$LOG_DIR" "$VIDEO_SAVE_DIR"

if [ ! -d "${CKPT_PATH}.net_model" ] || [ ! -f "${CKPT_PATH}.pth" ]; then
    echo "ERROR: incomplete checkpoint prefix: $CKPT_PATH" >&2
    exit 1
fi

if [ "$FORCE" = "1" ]; then
    echo "FORCE=1, removing previous outputs for $CKPT_STEP"
    rm -rf "$FINAL_VIDEO_DIR" "$HQ_VIDEO_DIR"
    rm -f "$LOG_DIR/infer_${CKPT_STEP}_student_8step_"*.log
fi

LOG_FILE="$LOG_DIR/infer_${CKPT_STEP}_student_8step_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date -Is)] START Wan2.1 DMD2 8-step inference"
echo "config=$CONFIG"
echo "ckpt_path=$CKPT_PATH"
echo "prompt_file=$PROMPT_FILE"
echo "negative_prompt_file=$NEG_PROMPT_FILE"
echo "video_save_dir=$VIDEO_SAVE_DIR"
echo "final_video_dir=$FINAL_VIDEO_DIR"
echo "log_file=$LOG_FILE"
echo "save_high_quality=$SAVE_HIGH_QUALITY"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

torchrun --standalone --nproc_per_node=1 \
    scripts/inference/video_model_inference.py \
    --config "$CONFIG" \
    --ckpt_path "$CKPT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --neg_prompt_file "$NEG_PROMPT_FILE" \
    --video_save_dir "$VIDEO_SAVE_DIR" \
    --do_student_sampling True \
    --do_teacher_sampling False \
    --save_high_quality "$SAVE_HIGH_QUALITY" \
    --save_as_gif False \
    --fps "$FPS" \
    - trainer.ddp=True \
      trainer.fsdp=False \
      trainer.fsdp_cpu_offload=False \
      trainer.batch_size_global=1 \
      dataloader_train.batch_size=1 \
      trainer.seed="$SEED" \
      log_config.name="wan21_t2v_dmd2_OpenVid_global_8_step8_infer_${CKPT_STEP}" \
    2>&1 | tee "$LOG_FILE"

status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "[$(date -Is)] FINISH status=$status"
    exit "$status"
fi

# The inference script appends "_hq" to the prompt directory when high-quality
# saving is enabled. Normalize it back to the standard eval path for report reuse.
if [ "$SAVE_HIGH_QUALITY" = "True" ] || [ "$SAVE_HIGH_QUALITY" = "true" ] || [ "$SAVE_HIGH_QUALITY" = "1" ]; then
    if [ -d "$HQ_VIDEO_DIR" ]; then
        rm -rf "$FINAL_VIDEO_DIR"
        mv "$HQ_VIDEO_DIR" "$FINAL_VIDEO_DIR"
        echo "Moved high-quality output $HQ_VIDEO_DIR -> $FINAL_VIDEO_DIR"
    fi
fi

echo "[$(date -Is)] FINISH status=0"
echo "Outputs under: $FINAL_VIDEO_DIR"
