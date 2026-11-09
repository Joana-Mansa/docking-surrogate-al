"""Device resolution.

Design note
-----------
Device selection happens exactly once, here, from the config value. No other
module calls ``torch.cuda.is_available()`` or constructs a device string. The
reason is that a silent CPU fallback is the single easiest way to produce a
result that looks fine and took forty times longer than it should have, or
that quietly used a different numeric path than the one being reported. So:

* ``cuda`` means *require* CUDA. If it is unavailable we raise, with a message
  that says what was requested, what torch sees, and how to proceed.
* ``cpu`` means CPU, even on a machine with a GPU.
* ``auto`` is the only mode permitted to fall back, and it logs that it did.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("cuda", "cpu", "auto")


class DeviceUnavailableError(RuntimeError):
    """Raised when the configured device cannot be used."""


def resolve_device(requested: str) -> torch.device:
    """Resolve a config device string to a concrete :class:`torch.device`.

    Args:
        requested: One of ``"cuda"``, ``"cpu"`` or ``"auto"``.

    Returns:
        The device to use for this run.

    Raises:
        DeviceUnavailableError: If ``"cuda"`` was requested and CUDA is not
            available, or if ``requested`` is not a recognised value.
    """
    value = requested.strip().lower()
    if value not in VALID_DEVICES:
        raise DeviceUnavailableError(
            f"run.device must be one of {VALID_DEVICES}, got {requested!r}"
        )

    if value == "cpu":
        logger.info("Device: CPU (explicitly requested).")
        return torch.device("cpu")

    available = torch.cuda.is_available()

    if value == "cuda":
        if not available:
            raise DeviceUnavailableError(
                "run.device is 'cuda' but torch reports no CUDA device.\n"
                f"  torch version:        {torch.__version__}\n"
                f"  torch CUDA build:     {torch.version.cuda or 'CPU-only build'}\n"
                f"  visible CUDA devices: {torch.cuda.device_count()}\n"
                "Either install the CUDA build of torch pinned in pyproject.toml "
                "(cu128), check that the NVIDIA driver is loaded (`nvidia-smi`), "
                "or set run.device to 'cpu' or 'auto'."
            )
        device = torch.device("cuda")
        logger.info("Device: CUDA (%s).", torch.cuda.get_device_name(device))
        return device

    # auto
    if available:
        device = torch.device("cuda")
        logger.info("Device: auto resolved to CUDA (%s).", torch.cuda.get_device_name(device))
        return device
    logger.warning(
        "Device: auto resolved to CPU; no CUDA device visible to torch "
        "(torch %s, CUDA build %s). Training will be slow.",
        torch.__version__,
        torch.version.cuda or "CPU-only build",
    )
    return torch.device("cpu")


def assert_device_available(requested: str) -> torch.device:
    """Resolve the device at startup so failures happen before any work.

    Called by every CLI entry point before data is loaded, so that a run on a
    machine without the requested accelerator fails in the first second rather
    than after a twenty-minute data preparation stage.
    """
    device = resolve_device(requested)
    if device.type == "cuda":
        # Touch the device so a driver/library mismatch surfaces here rather
        # than deep inside the first training step.
        torch.zeros(1, device=device)
    return device
