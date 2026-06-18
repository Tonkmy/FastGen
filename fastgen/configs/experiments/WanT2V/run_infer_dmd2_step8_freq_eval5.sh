#!/usr/bin/env bash
set -euo pipefail

# Evaluate selected Wan2.1-T2V DMD2 8-step freq-run checkpoints on the first
# five OpenVid eval prompts. Defaults to checkpoint 0000500 on GPU 0.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXP_NAME=${EXP_NAME:-wan21_t2v_dmd2_OpenVid_global_8_step8_freq}
EXP_ROOT=${EXP_ROOT:-FASTGEN_OUTPUT/fastgen/wan_dmd2/$EXP_NAME}
CKPT_ROOT=${CKPT_ROOT:-$EXP_ROOT/checkpoints}
CONFIG=${CONFIG:-fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py}
PROMPT_SOURCE=${PROMPT_SOURCE:-scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt}
PROMPT_FILE=${PROMPT_FILE:-scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts_first5.txt}
NEG_PROMPT_FILE=${NEG_PROMPT_FILE:-scripts/inference/prompts/negative_prompt.txt}
EVAL_ROOT=${EVAL_ROOT:-$EXP_ROOT/inference/eval_5prompts}
LOG_DIR=${LOG_DIR:-$EXP_ROOT/logs/eval_5prompts_step8_freq}
SAVE_HIGH_QUALITY=${SAVE_HIGH_QUALITY:-True}
FPS=${FPS:-16}
SEED=${SEED:-0}
FORCE=${FORCE:-0}

mkdir -p "$(dirname "$PROMPT_FILE")" "$EVAL_ROOT" "$LOG_DIR"

if [ ! -f "$PROMPT_FILE" ]; then
    head -n 5 "$PROMPT_SOURCE" > "$PROMPT_FILE"
fi

PROMPT_COUNT=$(python3 -c 'from pathlib import Path; import sys; print(sum(1 for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()))' "$PROMPT_FILE")
PROMPT_STEM=$(basename "$PROMPT_FILE" .txt)

if [ "$PROMPT_COUNT" -ne 5 ]; then
    echo "ERROR: expected exactly 5 prompts, found $PROMPT_COUNT in $PROMPT_FILE" >&2
    exit 1
fi

if [ -n "${CKPTS:-}" ]; then
    read -r -a CKPT_LIST <<< "$CKPTS"
else
    CKPT_LIST=(0000500)
fi

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

run_case() {
    local ckpt="$1"
    local ckpt_path="$CKPT_ROOT/$ckpt"
    local save_dir="$EVAL_ROOT/${ckpt}_student_8step"
    local final_dir="$save_dir/$PROMPT_STEM"
    local hq_dir="${final_dir}_hq"

    if [ ! -d "${ckpt_path}.net_model" ] || [ ! -f "${ckpt_path}.pth" ]; then
        echo "ERROR: incomplete checkpoint prefix: $ckpt_path" >&2
        exit 1
    fi

    if [ "$FORCE" = "1" ]; then
        rm -rf "$final_dir" "$hq_dir"
        rm -f "$LOG_DIR/${ckpt}_student_8step_freq_"*.log
    fi

    if [ "$FORCE" != "1" ] && [ -d "$final_dir" ]; then
        local existing
        existing=$(find "$final_dir" -maxdepth 1 -type f -name "*.mp4" | wc -l)
        if [ "$existing" -ge "$PROMPT_COUNT" ]; then
            echo "SKIP complete: $final_dir has $existing mp4 files"
            return 0
        fi
    fi

    mkdir -p "$save_dir"
    local log_file="$LOG_DIR/${ckpt}_student_8step_freq_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date -Is)] START $ckpt 8-step freq student inference"
    echo "config=$CONFIG"
    echo "ckpt_path=$ckpt_path"
    echo "prompt_file=$PROMPT_FILE"
    echo "negative_prompt_file=$NEG_PROMPT_FILE"
    echo "final_video_dir=$final_dir"
    echo "log_file=$log_file"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    torchrun --standalone --nproc_per_node=1 \
        scripts/inference/video_model_inference.py \
        --config "$CONFIG" \
        --ckpt_path "$ckpt_path" \
        --prompt_file "$PROMPT_FILE" \
        --neg_prompt_file "$NEG_PROMPT_FILE" \
        --video_save_dir "$save_dir" \
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
          log_config.name="${EXP_NAME}_eval5_${ckpt}" \
        2>&1 | tee "$log_file"

    local status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "[$(date -Is)] FAIL $ckpt status=$status"
        exit "$status"
    fi

    if [ "$SAVE_HIGH_QUALITY" = "True" ] || [ "$SAVE_HIGH_QUALITY" = "true" ] || [ "$SAVE_HIGH_QUALITY" = "1" ]; then
        if [ -d "$hq_dir" ]; then
            rm -rf "$final_dir"
            mv "$hq_dir" "$final_dir"
            echo "Moved high-quality output $hq_dir -> $final_dir"
        fi
    fi

    local count
    count=$(find "$final_dir" -maxdepth 1 -type f -name "*.mp4" | wc -l)
    echo "[$(date -Is)] DONE $ckpt, mp4=$count, output=$final_dir"
}

echo "Prompt file: $PROMPT_FILE"
echo "Prompt count: $PROMPT_COUNT"
echo "Checkpoint list: ${CKPT_LIST[*]}"
echo "Output root: $EVAL_ROOT"
echo "Log dir: $LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for ckpt in "${CKPT_LIST[@]}"; do
    run_case "$ckpt"
done

echo "[$(date -Is)] 8-step freq eval-5 inference complete."
