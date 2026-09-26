"""Minimal logging setup shared by scripts and library code."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once; call from script entry points only."""
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S")
