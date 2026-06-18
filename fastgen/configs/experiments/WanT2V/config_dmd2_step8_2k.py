# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from fastgen.configs.experiments.WanT2V.config_dmd2_smoke import create_config as base_create_config

"""Smoke-test config for DMD2 on Wan2.1-T2V-1.3B.

This config keeps the official WanT2V DMD2 recipe and only changes the
minimum settings needed for a short server-side wiring test.
"""


def create_config():
    config = base_create_config()

    # 8 denoising steps need 9 timestamps.
    # First run: interpolate current 4-step anchors, preserving the known schedule.
    config.model.student_sample_steps = 8
    config.model.sample_t_cfg.t_list = [
          0.999,
          0.968,
          0.937,
          0.885,
          0.833,
          0.729,
          0.624,
          0.312,
          0.0,
      ]
    # trainer loops roughly range(1, max_iter), so use 2001 if you want checkpoint at 2000.
    config.trainer.max_iter = 1000
    config.trainer.logging_iter = 20
    config.trainer.save_ckpt_iter = 500
    config.dataloader_train.batch_size = 1
    config.trainer.batch_size_global = 10
    config.model.student_update_freq = 2

    # Important: separate output dir from existing 4-step run.
    config.log_config.name = "wan21_t2v_dmd2_OpenVid_global_8_step8_freq"

    return config

# CUDA_VISIBLE_DEVICES=2,3,4,5,6 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=5 train.py \
# --config=fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py




# inference
# python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
#   --classes=10 --prompt_file=scripts/inference/prompts/classes.txt --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/debug/checkpoints/0002000.pth - log_config.name=test_inference