# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_dmd2 import create_config as dmd2_wan_create_config
from fastgen.configs.net import CKPT_ROOT_DIR

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

    # Keep the smoke test tiny: one sample per step, ten training iterations, one checkpoint.
    config.dataloader_train.batch_size = 1
    config.trainer.batch_size_global = 1
    config.trainer.max_iter = 10
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 10

    # Single-process smoke test. Use the official config as the base before scaling to FSDP.
    config.trainer.ddp = False
    config.trainer.fsdp = False

    config.log_config.name = "wan21_t2v_dmd2_smoke"

    return config
