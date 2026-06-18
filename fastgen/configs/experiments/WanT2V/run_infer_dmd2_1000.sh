#!/usr/bin/env bash
set -euo pipefail

# Student-only Wan2.1 DMD2 inference from the 1000-step FSDP checkpoint.
# Run from anywhere; this script cd's to the FastGen repo root.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CKPT_PATH=${CKPT_PATH:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/checkpoints/0001000}
PROMPT_FILE=${PROMPT_FILE:-scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt}
NEG_PROMPT_FILE=${NEG_PROMPT_FILE:-scripts/inference/prompts/negative_prompt.txt}
VIDEO_SAVE_DIR=${VIDEO_SAVE_DIR:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/inference/0001000_student}
LOG_DIR=${LOG_DIR:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs}
mkdir -p "$VIDEO_SAVE_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/infer_0001000_$(date +%Y%m%d_%H%M%S).log"
echo "Writing inference log to $LOG_FILE"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using checkpoint prefix: $CKPT_PATH"

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

torchrun --standalone --nproc_per_node=1 \
    scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanT2V/config_dmd2_smoke.py \
    --ckpt_path "$CKPT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --neg_prompt_file "$NEG_PROMPT_FILE" \
    --video_save_dir "$VIDEO_SAVE_DIR" \
    --do_student_sampling True \
    --do_teacher_sampling False \
    --save_high_quality False \
    --fps 16 \
    - trainer.ddp=True \
      trainer.fsdp=False \
      trainer.fsdp_cpu_offload=False \
      trainer.batch_size_global=1 \
      dataloader_train.batch_size=1 \
      log_config.name=wan21_t2v_dmd2_OpenVid_global_8_infer_1000 \
    2>&1 | tee "$LOG_FILE"
