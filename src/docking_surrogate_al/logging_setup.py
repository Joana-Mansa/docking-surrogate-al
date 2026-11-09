"""Logging configuration shared by every CLI entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging to stderr and, optionally, a file.

    The file handler is what makes an interrupted long run diagnosable after
    the fact: the console scrollback is gone, ``runs/<name>/run.log`` is not.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # RDKit is extremely chatty about sanitisation on real ChEMBL data.
    logging.getLogger("rdkit").setLevel(logging.ERROR)
