#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation for Wan2.1 T2V DMD2 checkpoints.
# Run from anywhere. Override CUDA_VISIBLE_DEVICES, CKPTS, FORCE, TEACHER_STEPS, or EVAL_ROOT as needed.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export EXP_ROOT=${EXP_ROOT:-FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8}
export CKPT_ROOT=${CKPT_ROOT:-$EXP_ROOT/checkpoints}
export PROMPT_FILE=${PROMPT_FILE:-scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt}
export NEG_PROMPT_FILE=${NEG_PROMPT_FILE:-scripts/inference/prompts/negative_prompt.txt}
export EVAL_ROOT=${EVAL_ROOT:-$EXP_ROOT/inference/eval_10prompts}
export LOG_DIR=${LOG_DIR:-$EXP_ROOT/logs/eval_10prompts}
export TEACHER_STEPS=${TEACHER_STEPS:-50}
export FPS=${FPS:-16}
export FORCE=${FORCE:-0}

mkdir -p "$EVAL_ROOT" "$LOG_DIR"

PROMPT_COUNT=$(python3 -c 'import os; from pathlib import Path; p=Path(os.environ["PROMPT_FILE"]); print(sum(1 for line in p.read_text().splitlines() if line.strip()))')
PROMPT_STEM=$(basename "$PROMPT_FILE" .txt)

echo "Prompt count: $PROMPT_COUNT"
echo "Prompt file: $PROMPT_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Output root: $EVAL_ROOT"
echo "Log dir: $LOG_DIR"

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

is_complete() {
    local save_dir="$1"
    local final_dir="$save_dir/$PROMPT_STEM"
    if [ "$FORCE" = "1" ]; then
        return 1
    fi
    if [ -d "$final_dir" ]; then
        local count
        count=$(find "$final_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)
        if [ "$count" -ge "$PROMPT_COUNT" ]; then
            echo "SKIP complete: $final_dir has $count mp4 files"
            return 0
        fi
        echo "INCOMPLETE: $final_dir has $count/$PROMPT_COUNT mp4 files"
    fi
    return 1
}

run_case() {
    local label="$1"
    local ckpt_path="$2"
    local save_dir="$3"
    local do_student="$4"
    local do_teacher="$5"
    local log_name="$6"

    if is_complete "$save_dir"; then
        return 0
    fi

    mkdir -p "$save_dir"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${label}_${stamp}.log"

    echo "[$(date -Is)] START $label"
    echo "  ckpt_path=$ckpt_path"
    echo "  save_dir=$save_dir"
    echo "  log_file=$log_file"

    torchrun --standalone --nproc_per_node=1 \
        scripts/inference/video_model_inference.py \
        --config fastgen/configs/experiments/WanT2V/config_dmd2_smoke.py \
        --ckpt_path "$ckpt_path" \
        --prompt_file "$PROMPT_FILE" \
        --neg_prompt_file "$NEG_PROMPT_FILE" \
        --video_save_dir "$save_dir" \
        --do_student_sampling "$do_student" \
        --do_teacher_sampling "$do_teacher" \
        --save_high_quality False \
        --fps "$FPS" \
        --num_steps "$TEACHER_STEPS" \
        - trainer.ddp=True \
          trainer.fsdp=False \
          trainer.fsdp_cpu_offload=False \
          trainer.batch_size_global=1 \
          dataloader_train.batch_size=1 \
          trainer.seed=0 \
          log_config.name="$log_name" \
        2>&1 | tee "$log_file"

    echo "[$(date -Is)] DONE $label"
}

run_case "teacher_steps${TEACHER_STEPS}" "" "$EVAL_ROOT/teacher_steps${TEACHER_STEPS}" False True "wan21_t2v_teacher_eval_${TEACHER_STEPS}steps"

if [ -n "${CKPTS:-}" ]; then
    read -r -a CKPT_LIST <<< "$CKPTS"
else
    CKPT_LIST=(0000500 0001000 0001500 0002000 0002500 0003000 0003500 0004000 0004500 0005000)
fi

for ckpt in "${CKPT_LIST[@]}"; do
    prefix="$CKPT_ROOT/$ckpt"
    if [ ! -d "${prefix}.net_model" ] && [ ! -f "${prefix}.pth" ]; then
        echo "WARN missing checkpoint prefix: $prefix"
        continue
    fi
    run_case "${ckpt}_student_step4" "$prefix" "$EVAL_ROOT/${ckpt}_student_step4" True False "wan21_t2v_dmd2_eval_${ckpt}"
done

echo "[$(date -Is)] Evaluation sweep complete."
