from __future__ import annotations

import base64
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from decode_nmt import Hypothesis


def get_diagnostic_dir() -> Path:
  """Create outputs/beam_search_diagnostics like the starter. :contentReference[oaicite:8]{index=8}"""
  p = Path.cwd() / "outputs" / "beam_search_diagnostics"
  p.mkdir(parents=True, exist_ok=True)
  return p


def get_diagnostic_info() -> str:
  """Return base64-encoded JSON blob with time/host/user."""
  d = {
    "t": datetime.utcnow().isoformat(),
    "h": socket.gethostname(),
    "u": os.getlogin(),
  }
  return base64.b64encode(json.dumps(d).encode("utf-8")).decode("utf-8")


def record_train_diagnostics(payload: Dict[str, Any], step: int) -> None:
  """Write a JSON file for a given training step."""
  out = get_diagnostic_dir() / f"{step:06}.json"
  out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_example_sentence(
  src_pieces: List[str],
  tgt_pieces: List[str],
  hyps: List[Hypothesis],
  step: int,
) -> None:
  """Record an example translation beam to disk."""
  payload = {
    "example_source": src_pieces,
    "example_target": tgt_pieces,
    "hypotheses": [{"hypothesis": h.token_ids, "score": h.score} for h in hyps],
    "diagnostic_info": get_diagnostic_info(),
  }
  record_train_diagnostics(payload, step)
