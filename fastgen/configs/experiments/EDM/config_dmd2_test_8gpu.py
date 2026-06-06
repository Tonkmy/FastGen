# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.EDM.config_dmd2_test import create_config as dmd2_test_create_config


def create_config():
    config = dmd2_test_create_config()

    # Eight-GPU DDP version of the EDM DMD2 test config.
    # Keep the same global batch size as config_dmd2_test.py: 8 GPUs * 8 samples/GPU = 64.
    config.trainer.ddp = True
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 1000
    config.dataloader_train.batch_size = 8
    config.trainer.batch_size_global = 64
    config.log_config.name = "dmd2_test_8gpu"
    config.log_config.wandb_mode = "disabled"

    return config
