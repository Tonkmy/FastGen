# FastGen Usage Notes

FastGen uses a generic training entry point with configuration-driven model and method selection.

## Training Entry Point

`train.py` is not specific to EDM. It is the common training entry point for all supported methods and networks.

At a high level, it does three things:

```python
config.model_class.config = config.model
model = instantiate(config.model_class)
trainer = Trainer(config)
trainer.run(model)
```

The selected config decides what is actually trained.

## Output Layout

By default, FastGen writes data, model checkpoints, and run outputs under the repository-local `FASTGEN_OUTPUT/` directory:

```text
FASTGEN_OUTPUT/
├── DATA/
├── MODEL/
└── fastgen/
```

This is controlled by the default config behavior when `FASTGEN_OUTPUT_ROOT`, `DATA_ROOT_DIR`, and `CKPT_ROOT_DIR` are not set. Keep these environment variables unset when you want the same layout as the upstream repository.

## Config Structure

FastGen configs are organized in two main layers:

```text
fastgen/configs/methods/
fastgen/configs/experiments/
```

Method configs define the training algorithm:

```text
fastgen/configs/methods/config_dmd2.py
fastgen/configs/methods/config_cm.py
fastgen/configs/methods/config_sft.py
fastgen/configs/methods/config_kd.py
```

Experiment configs apply a method to a concrete model, dataset, checkpoint, and set of hyperparameters:

```text
fastgen/configs/experiments/EDM/config_dmd2_test.py
fastgen/configs/experiments/SDXL/config_dmd2.py
fastgen/configs/experiments/WanT2V/config_dmd2.py
```

For example, `fastgen/configs/experiments/EDM/config_dmd2_test.py` starts from the DMD2 method config and then sets the EDM CIFAR-10 checkpoint:

```python
config = dmd2_create_config()
config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"
```

Other model families follow the same pattern. They do not need a different training script; they provide a different experiment config.

For example, the SDXL DMD2 config changes the network, data loader, input shape, discriminator, and training hyperparameters:

```python
config = config_dmd2_default.create_config()
config.model.net = SDXLConfig
config.dataloader_train = ImageLatentLoaderConfig
config.model.input_shape = [4, 128, 128]
config.model.discriminator = Discriminator_SDXL_Res1024_Config
```

The Wan text-to-video DMD2 config similarly changes the network and data loader:

```python
config = config_dmd2_default.create_config()
config.model.net = Wan_1_3B_Config
config.dataloader_train = VideoLoaderConfig
config.model.input_shape = [16, 21, 60, 104]
```

## Adapting Another Model

To adapt another model, prefer adding an experiment config instead of editing `train.py`.

The experiment config usually needs to specify:

- `config.model.net`: the network implementation config
- `config.model.input_shape`: the model input shape
- `config.model.pretrained_model_path`: the teacher or initialization checkpoint
- `config.dataloader_train`: the dataset loader config
- method-specific settings, such as discriminator, guidance scale, time sampling, or student steps
- trainer settings, such as batch size, max iteration count, logging interval, and checkpoint interval

If the existing method code can train the new network through the common FastGen network interface, only config changes are needed. If the model has different forward signatures, conditioning, preprocessing, or sampling behavior, the network wrapper under `fastgen/networks/` is the main adaptation point.

## Local CPU Smoke Config

This repository includes a local smoke-test config:

```text
fastgen/configs/experiments/EDM/config_dmd2_smoke_cpu.py
```

It inherits from:

```text
fastgen/configs/experiments/EDM/config_dmd2_test.py
```

and only changes the settings needed to validate the training loop on a machine without CUDA:

```python
config.model.device = "cpu"
config.trainer.max_iter = 2
config.trainer.batch_size_global = 2
config.dataloader_train.batch_size = 2
config.trainer.callbacks = DictConfig({})
config.log_config.wandb_mode = "disabled"
```

Run it with:

```bash
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_smoke_cpu.py
```

This config is for checking wiring only: config loading, checkpoint loading, dataloader construction, training-loop entry, and checkpoint saving. It is not intended for meaningful training quality.
