# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import os
import time
from numbers import Number
from typing import TYPE_CHECKING, Callable, Any

import psutil
import torch
import torch.distributed as dist

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import get_rank, is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


GB = 1024**3


class LocalStatsCallback(Callback):
    """Persist scalar losses, step time, and CUDA memory stats without W&B."""

    def __init__(
        self,
        every_n: int | None = None,
        log_dir: str = "logs",
        filename_prefix: str = "train_stats",
        synchronize_cuda: bool = True,
        include_rank_details: bool = True,
    ):
        self.every_n = every_n
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.synchronize_cuda = synchronize_cuda
        self.include_rank_details = include_rank_details

        self._step_start_time: float | None = None
        self._csv_path: str | None = None
        self._jsonl_path: str | None = None
        self._csv_fieldnames: list[str] = []
        self._csv_rows: list[dict[str, Any]] = []
        self._global_peak_allocated_gb = 0.0
        self._global_peak_reserved_gb = 0.0

    def on_app_begin(self) -> None:
        if self.every_n is None and hasattr(self, "config"):
            self.every_n = self.config.trainer.logging_iter
        if self.every_n is None:
            self.every_n = 100

        save_path = self.config.log_config.save_path if hasattr(self, "config") else os.getcwd()
        output_dir = os.path.join(save_path, self.log_dir)
        self._csv_path = os.path.join(output_dir, f"{self.filename_prefix}.csv")
        self._jsonl_path = os.path.join(output_dir, f"{self.filename_prefix}.jsonl")

        if is_rank0():
            os.makedirs(output_dir, exist_ok=True)
            logger.info(
                f"LocalStatsCallback writing every {self.every_n} iteration(s) to "
                f"{self._jsonl_path} and {self._csv_path}"
            )

    def on_training_step_begin(self, model: "FastGenModel", iteration: int = 0) -> None:
        del model
        if torch.cuda.is_available():
            if self.synchronize_cuda:
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._step_start_time = time.perf_counter()

    def on_training_step_end(
        self,
        model: "FastGenModel",
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del model, data_batch, output_batch
        should_log = iteration == 1 or iteration % self.every_n == 0

        if should_log and torch.cuda.is_available() and self.synchronize_cuda:
            torch.cuda.synchronize()

        elapsed_s = None
        if self._step_start_time is not None:
            elapsed_s = time.perf_counter() - self._step_start_time

        rank_stats = self._get_rank_stats(elapsed_s)
        losses = self._losses_to_float(loss_dict)

        self._global_peak_allocated_gb = max(
            self._global_peak_allocated_gb, rank_stats.get("cuda_step_peak_allocated_gb", 0.0)
        )
        self._global_peak_reserved_gb = max(
            self._global_peak_reserved_gb, rank_stats.get("cuda_step_peak_reserved_gb", 0.0)
        )
        rank_stats["cuda_global_peak_allocated_gb"] = self._global_peak_allocated_gb
        rank_stats["cuda_global_peak_reserved_gb"] = self._global_peak_reserved_gb

        if not should_log:
            return

        rank_payload = {"rank_stats": rank_stats, "losses": losses}
        gathered_payload = self._all_gather_object(rank_payload)

        if not is_rank0():
            return

        rank_records = [payload["rank_stats"] for payload in gathered_payload]
        loss_records = [payload["losses"] for payload in gathered_payload]
        avg_losses = self._average_losses(loss_records)
        summary = self._summarize_rank_stats(rank_records)

        record = {
            "iteration": iteration,
            "time_unix": time.time(),
            "world_size": len(rank_records),
            "losses": avg_losses,
            "summary": summary,
        }
        if self.include_rank_details:
            record["ranks"] = rank_records

        self._write_jsonl(record)
        self._write_csv(self._flatten_record(record))

        total_loss = avg_losses.get("total_loss")
        total_loss_text = f", total_loss={total_loss:.4f}" if total_loss is not None else ""
        peak_text = summary.get("cuda_step_peak_allocated_gb/max")
        peak_text = f", step_peak_allocated={peak_text:.2f}GB" if peak_text is not None else ""
        time_text = summary.get("step_time_s/max")
        time_text = f", step_time_max={time_text:.2f}s" if time_text is not None else ""
        logger.info(f"LocalStatsCallback logged iteration {iteration}{total_loss_text}{peak_text}{time_text}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_peak_allocated_gb": self._global_peak_allocated_gb,
            "global_peak_reserved_gb": self._global_peak_reserved_gb,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._global_peak_allocated_gb = float(state_dict.get("global_peak_allocated_gb", 0.0))
        self._global_peak_reserved_gb = float(state_dict.get("global_peak_reserved_gb", 0.0))

    def _get_rank_stats(self, elapsed_s: float | None) -> dict[str, float | int | None]:
        process = psutil.Process(os.getpid())
        cpu_rss_bytes = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                cpu_rss_bytes += child.memory_info().rss
            except psutil.Error:
                pass

        stats: dict[str, float | int | None] = {
            "rank": get_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "pid": os.getpid(),
            "step_time_s": elapsed_s,
            "cpu_rss_gb": cpu_rss_bytes / GB,
        }

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            stats.update(
                {
                    "cuda_device": device,
                    "cuda_current_allocated_gb": torch.cuda.memory_allocated(device) / GB,
                    "cuda_current_reserved_gb": torch.cuda.memory_reserved(device) / GB,
                    "cuda_step_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GB,
                    "cuda_step_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GB,
                }
            )

        return stats

    def _losses_to_float(self, loss_dict: dict[str, torch.Tensor] | None) -> dict[str, float]:
        if loss_dict is None:
            return {}

        losses = {}
        for key, value in loss_dict.items():
            try:
                if isinstance(value, torch.Tensor):
                    losses[key] = float(value.detach().float().mean().item())
                elif isinstance(value, Number):
                    losses[key] = float(value)
            except (RuntimeError, ValueError, TypeError):
                logger.warning(f"LocalStatsCallback could not serialize loss '{key}'")
        return losses

    def _all_gather_object(self, value: Any) -> list[Any]:
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, value)
            return gathered
        return [value]

    def _average_losses(self, loss_records: list[dict[str, float]]) -> dict[str, float]:
        names = sorted({name for losses in loss_records for name in losses})
        averaged = {}
        for name in names:
            values = [losses[name] for losses in loss_records if name in losses]
            if values:
                averaged[name] = sum(values) / len(values)
        return averaged

    def _summarize_rank_stats(self, rank_records: list[dict[str, float | int | None]]) -> dict[str, float]:
        names = sorted({name for record in rank_records for name in record})
        summary = {}
        for name in names:
            if name in {"rank", "local_rank", "pid", "cuda_device"}:
                continue
            values = [record[name] for record in rank_records if isinstance(record.get(name), Number)]
            if not values:
                continue
            summary[f"{name}/avg"] = sum(values) / len(values)
            summary[f"{name}/min"] = min(values)
            summary[f"{name}/max"] = max(values)
        return summary

    def _flatten_record(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "iteration": record["iteration"],
            "time_unix": record["time_unix"],
            "world_size": record["world_size"],
        }
        row.update({f"loss/{key}": value for key, value in record["losses"].items()})
        row.update(record["summary"])
        return row

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        assert self._jsonl_path is not None
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_csv(self, row: dict[str, Any]) -> None:
        assert self._csv_path is not None

        new_fields = [field for field in row if field not in self._csv_fieldnames]
        if new_fields:
            self._csv_fieldnames.extend(new_fields)
        self._csv_rows.append(row)

        with open(self._csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            writer.writerows(self._csv_rows)
