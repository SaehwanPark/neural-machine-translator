from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


def get_data_dir() -> Path:
  """Return the data directory from DATA_DIR (dotenv), defaulting to ~/data/machine_translation."""
  load_dotenv()
  default_dir = Path.home() / "data" / "machine_translation"
  return Path(os.getenv("DATA_DIR", str(default_dir))).expanduser().resolve()
