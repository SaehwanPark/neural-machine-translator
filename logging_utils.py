from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoggingConfig:
  """Configuration for application logging."""

  level: str = "INFO"
  log_file: Optional[Path] = None
  use_tqdm_handler: bool = False


def _parse_level(level: str) -> int:
  """Parse a string logging level into a logging level int."""
  return getattr(logging, level.upper(), logging.INFO)


def configure_logging(cfg: LoggingConfig) -> None:
  """
  Configure root logging handlers/format.

  Notes:
    - Call this exactly once in the CLI entrypoint (not in library modules).
    - Library modules should only do: logger = logging.getLogger(__name__).
  """
  level = _parse_level(cfg.level)

  root = logging.getLogger()
  root.setLevel(level)

  # Remove any pre-existing handlers (important in notebooks/tests).
  for h in list(root.handlers):
    root.removeHandler(h)

  fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
  formatter = logging.Formatter(fmt=fmt)

  stream_handler: logging.Handler
  if cfg.use_tqdm_handler:
    stream_handler = _TqdmLoggingHandler()
  else:
    stream_handler = logging.StreamHandler()

  stream_handler.setLevel(level)
  stream_handler.setFormatter(formatter)
  root.addHandler(stream_handler)

  if cfg.log_file is not None:
    cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(cfg.log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

  # Quiet overly-chatty third-party loggers by default.
  logging.getLogger("jax").setLevel(max(level, logging.WARNING))
  logging.getLogger("flax").setLevel(max(level, logging.WARNING))


class _TqdmLoggingHandler(logging.Handler):
  """A logging handler that plays nicely with tqdm progress bars."""

  def emit(self, record: logging.LogRecord) -> None:
    try:
      from tqdm import tqdm

      msg = self.format(record)
      tqdm.write(msg)
    except Exception:
      # Fallback to stderr if tqdm isn't available or something goes wrong.
      import sys

      sys.stderr.write(self.format(record) + "\n")
