#!/usr/bin/env bash
set -euo pipefail

# Encode OpenVid raw mp4/txt WebDataset into Wan2.1 latent/text-embedding WDS.
# The output is directly consumable by fastgen.configs.data.VideoLatentLoaderConfig.

REPO_ROOT=/data/chenqingzhan/FastGen
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-FASTGEN_OUTPUT}
export HF_HOME=${HF_HOME:-/data/chenqingzhan/.cache/huggingface}
export LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

INPUT_DIR=${INPUT_DIR:-/data/datasets/OpenVid-1M/webdataset}
OUTPUT_DIR=${OUTPUT_DIR:-/data/datasets/OpenVid-1M/wan21_latents_81x480x832_wds}
MODEL_PATH=${MODEL_PATH:-FASTGEN_OUTPUT/MODEL/Wan-AI/Wan2.1-T2V-1.3B-Diffusers}
NEG_PROMPT_FILE=${NEG_PROMPT_FILE:-scripts/inference/prompts/negative_prompt.txt}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/encode_$(date +%Y%m%d_%H%M%S).log"

echo "Writing encode log to $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODEL_PATH=$MODEL_PATH"

source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh
conda activate fastgen

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    scripts/data/encode_openvid_wan21_latents.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --neg_prompt_file "$NEG_PROMPT_FILE" \
    --sequence_length 81 \
    --width 832 \
    --height 480 \
    --precision bfloat16 \
    --storage_dtype bfloat16 \
    --log_every 25 \
    2>&1 | tee "$LOG_FILE"
