# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_dmd2_smoke import create_config as base_create_config

"""Progressive DMD2 stage: initialize 4-step Wan2.1-T2V student from a 2500-iter 8-step student.

This stage should be launched only after the previous stage was truly trained with
model.student_sample_steps = 8. The run shell script validates that before launch.
"""


def create_config():
    config = base_create_config()

    # Target stage: 4-step student.
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Train this progressive stage for 2500 iterations and keep 500-iter checkpoints.
    config.trainer.max_iter = 1000
    config.trainer.logging_iter = 20
    config.trainer.save_ckpt_iter = 500
    config.dataloader_train.batch_size = 1
    config.trainer.batch_size_global = 10

    # Start a new run. Do not auto-resume from this new output directory.
    config.trainer.resume = False

    # Load the student net from the previous 8-step FSDP checkpoint.
    # This path is a checkpoint prefix, not a .pth file.
    config.trainer.checkpointer.pretrained_ckpt_path = (
        "FASTGEN_OUTPUT/fastgen/wan_dmd2/"
        "wan21_t2v_dmd2_OpenVid_global_8_step8_2k/checkpoints/0002500"
    )
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    # Keep this empty. It expects a single ordinary torch state_dict file, while the
    # previous stage checkpoint is an FSDP checkpoint prefix loaded through trainer.checkpointer.
    config.model.pretrained_student_net_path = ""

    # Separate output directory for this progressive stage.
    config.log_config.name = "wan21_t2v_dmd2_OpenVid_global_8_step4_from_step8_2p5k"

    return config
