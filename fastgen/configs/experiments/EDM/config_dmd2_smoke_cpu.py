# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.configs.experiments.EDM.config_dmd2_test import create_config as dmd2_test_create_config


def create_config():
    config = dmd2_test_create_config()

    # This config is only for a local CPU smoke test on machines without CUDA.
    # It verifies config/model/data/training-loop wiring, not training quality.
    config.model.device = "cpu"
    config.trainer.max_iter = 2
    config.trainer.batch_size_global = 2
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 1000
    config.trainer.resume = False
    config.trainer.callbacks = DictConfig({})
    config.dataloader_train.batch_size = 2
    config.log_config.name = "smoke_test_cpu"
    config.log_config.wandb_mode = "disabled"

    return config
