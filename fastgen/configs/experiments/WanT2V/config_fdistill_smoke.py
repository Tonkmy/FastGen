# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_fdistill import create_config as fdistill_wan_create_config
from fastgen.configs.net import CKPT_ROOT_DIR
from fastgen.configs.callbacks import LocalStats_CALLBACK

"""Smoke-test config for f-distill on Wan2.1-T2V-1.3B with OpenVid.

This keeps the official WanT2V f-distill recipe and only changes the wiring
needed for our local Diffusers teacher, OpenVid WebDataset, FSDP launch, and
lightweight local logging.
"""


def create_config():
    config = fdistill_wan_create_config()

    # Use the repo-local Diffusers teacher/student source.
    config.model.net.model_id_or_local_path = f"{CKPT_ROOT_DIR}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # OpenVid is already in raw mp4/txt WebDataset format.
    config.dataloader_train.datatags = ["WDS:/data/datasets/OpenVid-1M/webdataset"]
    config.dataloader_train.batch_size = 2

    # Match the DMD2 OpenVid wiring: one sample per GPU, global batch 8,
    # periodic local stats, and sparse checkpoints.
    config.trainer.max_iter = 1000
    config.trainer.logging_iter = 9
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 16

    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = False

    # Disable W&B media logging for Wan video. Keep local scalar/timing/memory logs.
    config.trainer.callbacks.pop("wandb", None)
    config.trainer.callbacks.update(LocalStats_CALLBACK)

    config.log_config.name = "wan21_t2v_fdistill_OpenVid_global_16"

    return config


# 8-GPU launch, preferred because f-distill has DMD2-like net/fake_score/discriminator state:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=8 train.py \
#     --config=fastgen/configs/experiments/WanT2V/config_fdistill_smoke.py
#
# 4-GPU fallback for physical GPUs 4-7. CPU offload is recommended if memory is tight:
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=4 train.py \
#     --config=fastgen/configs/experiments/WanT2V/config_fdistill_smoke.py \

#     - trainer.fsdp_cpu_offload=True
