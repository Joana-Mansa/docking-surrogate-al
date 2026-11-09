"""Hardware introspection and throughput accounting.

Design note
-----------
Timings without hardware context are not interpretable: "trained in 4 minutes"
means nothing unless the manifest also says on what. This module collects the
facts that make a wall-clock number comparable across machines, and records
them in the run manifest next to the CPU timings:

* GPU name, total VRAM, driver and CUDA runtime version
* peak allocated and peak reserved VRAM for the stage
* throughput in samples per second

Peak memory is read from torch's allocator rather than ``nvidia-smi``, so it
reflects this process only and is not polluted by other GPU users.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from time import perf_counter
from types import TracebackType
from typing import Any

import torch


@dataclass(frozen=True)
class HardwareInfo:
    """Static description of the machine a stage ran on."""

    platform: str
    python_version: str
    cpu_count: int | None
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cudnn_version: int | None
    device_type: str
    gpu_name: str | None = None
    gpu_total_vram_mb: float | None = None
    gpu_capability: str | None = None
    gpu_driver_version: str | None = None
    gpu_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


def collect_hardware_info(device: torch.device) -> HardwareInfo:
    """Gather hardware facts for the run manifest."""
    gpu_name: str | None = None
    vram_mb: float | None = None
    capability: str | None = None
    driver: str | None = None
    gpu_count = 0

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        index = device.index or 0 if device.type == "cuda" else 0
        props = torch.cuda.get_device_properties(index)
        gpu_name = props.name
        vram_mb = round(props.total_memory / 1024**2, 1)
        capability = f"{props.major}.{props.minor}"
        # Only available on recent torch builds; absence is not an error.
        driver = getattr(torch.cuda, "driver_version", lambda: None)()
        driver = str(driver) if driver is not None else None

    return HardwareInfo(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        device_type=device.type,
        gpu_name=gpu_name,
        gpu_total_vram_mb=vram_mb,
        gpu_capability=capability,
        gpu_driver_version=driver,
        gpu_count=gpu_count,
    )


def reset_peak_memory(device: torch.device) -> None:
    """Reset torch's peak-memory counters for ``device``."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> dict[str, float | None]:
    """Return peak allocated/reserved VRAM in MiB since the last reset."""
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }


class StageTimer:
    """Context manager measuring wall-clock time, throughput and peak VRAM.

    Example:
        >>> with StageTimer("train", device) as t:   # doctest: +SKIP
        ...     run_training()
        ...     t.add_samples(n_graphs_seen)
        >>> t.summary()                               # doctest: +SKIP
    """

    def __init__(self, stage: str, device: torch.device) -> None:
        self.stage = stage
        self.device = device
        self.samples = 0
        self._start = 0.0
        self.elapsed_s = 0.0

    def __enter__(self) -> StageTimer:
        reset_peak_memory(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_s = perf_counter() - self._start

    def add_samples(self, n: int) -> None:
        """Record ``n`` processed samples for the throughput figure."""
        self.samples += n

    def summary(self) -> dict[str, Any]:
        """Return the manifest entry for this stage."""
        out: dict[str, Any] = {
            "stage": self.stage,
            "wall_clock_s": round(self.elapsed_s, 3),
            "samples": self.samples,
            "samples_per_second": (
                round(self.samples / self.elapsed_s, 2) if self.elapsed_s > 0 and self.samples else None
            ),
            "device": str(self.device),
        }
        out.update(peak_memory_mb(self.device))
        return out
