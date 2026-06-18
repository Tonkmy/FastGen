# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_mf import create_config as mf_wan_create_config
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CKPT_ROOT_DIR
from fastgen.configs.callbacks import LocalStats_CALLBACK

"""Smoke-test config for MeanFlow on Wan2.1-T2V-1.3B with OpenVid latents.

This keeps the official WanT2V MeanFlow recipe and only changes the wiring
needed for our local Diffusers teacher, pre-encoded OpenVid latent WebDataset,
FSDP launch, and lightweight local logging.
"""


def create_config():
    config = mf_wan_create_config()

    # Use the repo-local Diffusers teacher/student source.
    config.model.net.model_id_or_local_path = f"{CKPT_ROOT_DIR}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # OpenVid is pre-encoded as latent.pth/txt_emb.pth WebDataset shards by
    # scripts/data/encode_openvid_wan21_latents.py.
    config.model.enable_preprocessors = False
    config.model.input_shape = [16, 21, 60, 104]
    config.dataloader_train = VideoLatentLoaderConfig
    config.dataloader_train.datatags = ["WDS:/data/datasets/OpenVid-1M/wan21_latents_81x480x832_wds"]
    config.dataloader_train.files_map = {
        "neg_condition": "/data/datasets/OpenVid-1M/wan21_latents_81x480x832_wds/neg_prompt_emb.npy",
    }
    config.dataloader_train.batch_size = 2
    config.dataloader_train.shuffle_size = 32
    config.dataloader_train.num_workers = 0
    config.dataloader_train.pop("img_size", None)
    config.dataloader_train.pop("sequence_length", None)

    # Match the DMD2 OpenVid smoke/full-run wiring: one sample per GPU,
    # global batch 8, periodic local stats, and sparse checkpoints.
    config.trainer.max_iter = 2000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 8
    config.model.net_optimizer.lr = 1.25e-6

    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = False

    # Disable W&B media logging for Wan video. Keep local scalar/timing/memory logs.
    config.trainer.callbacks.pop("wandb", None)
    config.trainer.callbacks.update(LocalStats_CALLBACK)

    config.log_config.name = "wan21_t2v_mf_openvid_wan21_latents_global_8"

    return config


# 8-GPU launch, preferred when all cards are free:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=8 train.py \
#     --config=fastgen/configs/experiments/WanT2V/config_mf_smoke.py
#
# 4-GPU launch for the currently free physical GPUs 4-7:
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=4 train.py \
#     --config=fastgen/configs/experiments/WanT2V/config_mf_smoke.py
