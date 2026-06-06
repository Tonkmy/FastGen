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

## Atomic Workflow

FastGen has a clear three-stage workflow:

```text
data/model assets -> training run -> inference run
```

Each stage has a small set of inputs and outputs. This makes the repository useful as a reference for organizing reproducible research code.

### 1. Data And Model Assets

The data preparation script is:

```bash
python scripts/download_data.py --dataset cifar10
```

For the EDM CIFAR-10 workflow, it creates:

```text
FASTGEN_OUTPUT/DATA/cifar10/cifar10-32x32.zip
FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-uncond-vp.pth
FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-cond-vp.pth
```

These files are upstream assets, not training outputs from the current run:

- `DATA/.../cifar10-32x32.zip`: dataset consumed by the dataloader.
- `MODEL/.../edm-cifar10-32x32-cond-vp.pth`: pretrained teacher diffusion model used by DMD2.
- `MODEL/.../edm-cifar10-32x32-uncond-vp.pth`: pretrained unconditional EDM model for workflows that need it.

The zip dataset is intentionally left zipped. The data loader reads it directly.

### 2. Training Run

Training enters through:

```bash
python train.py --config=<experiment_config.py>
```

or for multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 train.py --config=<experiment_config.py>
```

The config determines the method, network, dataset, teacher checkpoint, optimizer, trainer behavior, and output location.

For example:

```text
fastgen/configs/experiments/EDM/config_dmd2_test.py
```

depends on:

```text
fastgen/configs/methods/config_dmd2.py
FASTGEN_OUTPUT/DATA/cifar10/cifar10-32x32.zip
FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-cond-vp.pth
```

and produces:

```text
FASTGEN_OUTPUT/fastgen/cifar10/<run_name>/
├── config.yaml
├── checkpoints/
│   ├── 0001000.pth
│   ├── 0002000.pth
│   └── ...
└── wandb/ or wandb_id.txt, depending on logging mode
```

`config.yaml` is the resolved training configuration. It is the record of what actually ran after inheritance and command-line overrides.

The checkpoint files are the primary training artifacts. They are later consumed by inference and evaluation scripts.

### 3. Inference Run

Image inference enters through:

```bash
python scripts/inference/image_model_inference.py \
  --config=<experiment_config.py> \
  --ckpt_path=<training_checkpoint.pth> \
  [task-specific args] \
  - log_config.name=<inference_run_name>
```

For the EDM CIFAR-10 DMD2 workflow:

```bash
python scripts/inference/image_model_inference.py \
  --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
  --classes=10 \
  --prompt_file=scripts/inference/prompts/classes.txt \
  --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/dmd2_test_8gpu/checkpoints/0005000.pth \
  - log_config.name=test_inference_8gpu log_config.wandb_mode=disabled
```

The inference config should describe the model architecture and task. It does not need to be the same config used to launch multi-GPU training. For example, if the training config only adds `trainer.ddp=True`, use the base model config for single-process inference and point `--ckpt_path` to the multi-GPU checkpoint.

The inference run produces samples under:

```text
FASTGEN_OUTPUT/fastgen/cifar10/<inference_run_name>/samples/
```

## Dependency Graph

For EDM CIFAR-10 DMD2, the core dependency graph is:

```text
scripts/download_data.py
  -> FASTGEN_OUTPUT/DATA/cifar10/cifar10-32x32.zip
  -> FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-cond-vp.pth

fastgen/configs/methods/config_dmd2.py
  -> DMD2 method: student net, fake score, discriminator, losses, callbacks

fastgen/configs/experiments/EDM/config_dmd2_test.py
  -> selects DMD2
  -> selects EDM CIFAR-10 teacher checkpoint
  -> sets training length, batch size, logging, checkpoint interval

train.py + config_dmd2_test.py
  -> FASTGEN_OUTPUT/fastgen/cifar10/<training_run>/checkpoints/*.pth

scripts/inference/image_model_inference.py + model config + checkpoint
  -> FASTGEN_OUTPUT/fastgen/cifar10/<inference_run>/samples/
```

The important pattern is that configs are the dependency contract. Scripts are generic entry points; configs bind a method, network, data source, and artifact path into a concrete experiment.

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
