#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Pre-encode OpenVid WebDataset videos for Wan2.1 latent training.

The output format is directly consumable by fastgen.configs.data.VideoLatentLoaderConfig:
each output shard contains sample keys with:

    <key>.latent.pth
    <key>.txt_emb.pth

and output_dir/neg_prompt_emb.npy stores the shared negative prompt embedding.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds

from fastgen.datasets.decoders import decode_video_segment
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.networks.Wan.network import WanTextEncoder, WanVideoEncoder


def _rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _tensor_to_pth_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _find_video_bytes(item: dict[str, Any]) -> bytes | None:
    for key in ("mp4", "webm", "mov", "avi"):
        value = item.get(key)
        if value is not None:
            return value
    return None


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _write_negative_prompt(
    output_dir: Path,
    text_encoder: WanTextEncoder,
    neg_prompt_file: Path,
    device: torch.device,
    precision: torch.dtype,
) -> None:
    if not neg_prompt_file.is_file():
        raise FileNotFoundError(f"Negative prompt file not found: {neg_prompt_file}")
    neg_prompt = neg_prompt_file.read_text().strip()
    with torch.inference_mode():
        neg = text_encoder.encode([neg_prompt], precision=precision)
    # Save without the batch dimension so DataLoader collation yields [B, 512, 4096].
    np.save(output_dir / "neg_prompt_emb.npy", neg[0].float().cpu().numpy())


def _load_models(model_path: str, device: torch.device, precision: torch.dtype) -> tuple[WanTextEncoder, WanVideoEncoder]:
    text_encoder = WanTextEncoder(model_id_or_local_path=model_path).to(device=device, dtype=precision)
    vae = WanVideoEncoder(model_id_or_local_path=model_path).to(device=device, dtype=precision)
    return text_encoder, vae


def _open_writer(path: Path) -> wds.TarWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return wds.TarWriter(str(path))


def encode_source_shard(
    *,
    src_idx: int,
    src_path: Path,
    output_dir: Path,
    text_encoder: WanTextEncoder,
    vae: WanVideoEncoder,
    device: torch.device,
    precision: torch.dtype,
    storage_dtype: torch.dtype,
    sequence_length: int,
    width: int,
    height: int,
    seed: int,
    overwrite: bool,
    log_every: int,
    max_items_per_shard: int | None,
) -> dict[str, Any]:
    final_path = output_dir / f"{src_idx:06d}.tar"
    tmp_path = output_dir / f"{src_idx:06d}.tar.tmp"
    if final_path.exists() and not overwrite:
        return {
            "source": str(src_path),
            "output": str(final_path),
            "status": "skipped_existing",
            "written": None,
            "skipped": None,
            "seconds": 0.0,
        }
    if tmp_path.exists():
        tmp_path.unlink()
    if final_path.exists() and overwrite:
        final_path.unlink()

    started = time.time()
    written = 0
    skipped = 0
    dataset = wds.WebDataset(
        [str(src_path)],
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        handler=wds.handlers.warn_and_continue,
    )

    with _open_writer(tmp_path) as sink:
        for item_idx, item in enumerate(dataset):
            if max_items_per_shard is not None and item_idx >= max_items_per_shard:
                break
            video_bytes = _find_video_bytes(item)
            text = _decode_text(item.get("txt", ""))
            if video_bytes is None or not text:
                skipped += 1
                continue

            try:
                random.seed(seed + src_idx * 1_000_000 + item_idx)
                frames = decode_video_segment("sample.mp4", video_bytes, num_frames=sequence_length, output_format="torch")
                if frames is None or frames.shape[0] < sequence_length:
                    skipped += 1
                    continue
                video = transform_video(frames, sequence_length=sequence_length, img_size=(width, height))["real"]
                video = video.unsqueeze(0).to(device=device, dtype=precision)

                with torch.inference_mode():
                    latent = vae.encode(video, mode="sample")[0].to(dtype=storage_dtype).cpu()
                    txt_emb = text_encoder.encode([text], precision=precision)[0].to(dtype=storage_dtype).cpu()

                out_key = f"{src_idx:06d}_{item_idx:06d}"
                sink.write(
                    {
                        "__key__": out_key,
                        "latent.pth": _tensor_to_pth_bytes(latent),
                        "txt_emb.pth": _tensor_to_pth_bytes(txt_emb),
                    }
                )
                written += 1
                if log_every > 0 and written % log_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"[shard {src_idx:06d}] written={written} skipped={skipped} "
                        f"elapsed={elapsed/60:.1f}m rate={written/max(elapsed, 1e-6):.3f}/s",
                        flush=True,
                    )
            except Exception as exc:
                skipped += 1
                print(f"[shard {src_idx:06d}] skipped item {item_idx}: {type(exc).__name__}: {exc}", flush=True)

    tmp_path.rename(final_path)
    seconds = time.time() - started
    return {
        "source": str(src_path),
        "output": str(final_path),
        "status": "done",
        "written": written,
        "skipped": skipped,
        "seconds": seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="/data/datasets/OpenVid-1M/webdataset")
    parser.add_argument("--output_dir", default="/data/datasets/OpenVid-1M/wan21_latents_81x480x832_wds")
    parser.add_argument("--model_path", default="FASTGEN_OUTPUT/MODEL/Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--neg_prompt_file", default="scripts/inference/prompts/negative_prompt.txt")
    parser.add_argument("--sequence_length", type=int, default=81)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--storage_dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--max_items_per_shard", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = _rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    precision = getattr(torch, args.precision)
    storage_dtype = getattr(torch, args.storage_dtype)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_shards = sorted(input_dir.glob("*.tar"))
    if args.max_shards is not None:
        source_shards = source_shards[: args.max_shards]
    if not source_shards:
        raise FileNotFoundError(f"No .tar shards found in {input_dir}")

    assigned = [(idx, path) for idx, path in enumerate(source_shards) if idx % world_size == rank]
    print(
        f"rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"assigned_shards={len(assigned)} output_dir={output_dir}",
        flush=True,
    )

    text_encoder, vae = _load_models(args.model_path, device=device, precision=precision)
    if rank == 0:
        _write_negative_prompt(output_dir, text_encoder, Path(args.neg_prompt_file), device, precision)
        metadata = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "model_path": args.model_path,
            "sequence_length": args.sequence_length,
            "width": args.width,
            "height": args.height,
            "latent_shape": [16, 21, 60, 104],
            "txt_emb_shape": [512, 4096],
            "storage_dtype": args.storage_dtype,
            "world_size": world_size,
            "source_shards": len(source_shards),
            "created_unix": time.time(),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rank_summary_path = output_dir / f"rank{rank:03d}_summary.jsonl"
    totals = {"written": 0, "skipped": 0, "seconds": 0.0}
    with rank_summary_path.open("a") as summary:
        for src_idx, src_path in assigned:
            result = encode_source_shard(
                src_idx=src_idx,
                src_path=src_path,
                output_dir=output_dir,
                text_encoder=text_encoder,
                vae=vae,
                device=device,
                precision=precision,
                storage_dtype=storage_dtype,
                sequence_length=args.sequence_length,
                width=args.width,
                height=args.height,
                seed=args.seed,
                overwrite=args.overwrite,
                log_every=args.log_every,
                max_items_per_shard=args.max_items_per_shard,
            )
            summary.write(json.dumps(result) + "\n")
            summary.flush()
            if result["written"] is not None:
                totals["written"] += result["written"]
            if result["skipped"] is not None:
                totals["skipped"] += result["skipped"]
            totals["seconds"] += result["seconds"]
            print(f"rank={rank} shard={src_idx:06d} result={result}", flush=True)

    print(f"rank={rank} complete totals={totals}", flush=True)


if __name__ == "__main__":
    main()
