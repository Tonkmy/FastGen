# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from fastgen.configs.callbacks import GradClip_CALLBACK, ParamCount_CALLBACK, TrainProfiler_CALLBACK
from fastgen.configs.experiments.EDM.config_dmd2_test import create_config as dmd2_test_create_config


def create_config():
    config = dmd2_test_create_config()

    # CPU version of the full EDM DMD2 test config.
    # It keeps the original test schedule and checkpoint, but removes CUDA/W&B callbacks.
    config.model.device = "cpu"
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
        }
    )
    config.log_config.name = "test_cpu"
    config.log_config.wandb_mode = "disabled"

    return config
