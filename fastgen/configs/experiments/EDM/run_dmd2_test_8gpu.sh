#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"



# watch -n 2 nvidia-smi


  # cd /data/chenqingzhan/FastGen

  # ps -ef | grep -E "torchrun|train.py" | grep -v grep

  # nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

  # ls -lh FASTGEN_OUTPUT/fastgen/cifar10/dmd2_test_8gpu/checkpoints

torchrun --nproc_per_node=8 train.py \
    --config=fastgen/configs/experiments/EDM/config_dmd2_test.py \
    - trainer.ddp=True log_config.name=test_ddp
# trainer.fsdp=True for large model

python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
  --classes=10 --prompt_file=scripts/inference/prompts/classes.txt --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/dmd2_test_8gpu/checkpoints/0005000.pth - log_config.name=test_inference_8gpu


torchrun --standalone --nproc_per_node=8 train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2_smoke.py