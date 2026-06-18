# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_dmd2 import create_config as dmd2_wan_create_config
from fastgen.configs.net import CKPT_ROOT_DIR
from fastgen.configs.callbacks import LocalStats_CALLBACK

"""Smoke-test config for DMD2 on Wan2.1-T2V-1.3B.

This config keeps the official WanT2V DMD2 recipe and only changes the
minimum settings needed for a short server-side wiring test.
"""


def create_config():
    config = dmd2_wan_create_config()

    # Use the repo-local Diffusers teacher we downloaded under FASTGEN_OUTPUT/MODEL.
    # This path contains transformer/, text_encoder/, tokenizer/, vae/, and scheduler/.
    config.model.net.model_id_or_local_path = f"{CKPT_ROOT_DIR}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # OpenVid is already in WebDataset format: each sample has an .mp4 video and a .txt caption.
    config.dataloader_train.datatags = ["WDS:/data/datasets/OpenVid-1M/webdataset"]

    # Five-step smoke: trainer loops range(1, max_iter), so max_iter=6 runs iterations 1..5.
    # Keep save_ckpt_iter larger than max_iter to avoid intermediate checkpoint spam.
    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 20
    config.trainer.save_ckpt_iter = 500
    config.dataloader_train.batch_size = 1
    config.trainer.batch_size_global = 10

    # Single-process smoke test. Use the official config as the base before scaling to FSDP.
    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_cpu_offload = False

    # Disable W&B for smoke: the default callback encodes 81-frame videos at iteration 1,
    # which can stall training for a long time before we know whether the step works.

    config.trainer.callbacks.pop("wandb", None)
    config.trainer.callbacks.update(LocalStats_CALLBACK)

    config.log_config.name = "wan21_t2v_dmd2_OpenVid_global_8"

    return config





# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HF_HOME=/data/chenqingzhan/.cache/huggingface \
# LOCAL_FILES_ONLY=true \
# torchrun --standalone --nproc_per_node=8 train.py \
#     --config=fastgen/configs/experiments/WanT2V/config_dmd2_smoke.py




# inference
# python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
#   --classes=10 --prompt_file=scripts/inference/prompts/classes.txt --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/debug/checkpoints/0002000.pth - log_config.name=test_inference