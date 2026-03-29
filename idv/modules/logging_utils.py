"""Structured JSON logging setup."""
from __future__ import annotations

import logging
from pathlib import Path

from pythonjsonlogger import jsonlogger


def setup_logger(log_path: Path) -> logging.Logger:
    """Create run-specific structured logger."""
    logger = logging.getLogger(f"idv.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
