from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sentencepiece as spm


@dataclass(frozen=True)
class SpmConfig:
  """SentencePiece training configuration."""

  vocab_size: int
  model_prefix: str
  character_coverage: float = 1.0
  model_type: str = "unigram"


def train_sentencepiece(input_path: Path, out_dir: Path, cfg: SpmConfig) -> Path:
  """Train a SentencePiece model and return the .model path."""
  out_dir.mkdir(parents=True, exist_ok=True)
  prefix_path = out_dir / cfg.model_prefix
  spm.SentencePieceTrainer.Train(
    input=str(input_path),
    model_prefix=str(prefix_path),
    vocab_size=int(cfg.vocab_size),
    character_coverage=float(cfg.character_coverage),
    model_type=str(cfg.model_type),
  )
  return Path(f"{prefix_path}.model")


def load_sp(model_path: Path) -> spm.SentencePieceProcessor:
  """Load a SentencePieceProcessor from a .model file."""
  sp = spm.SentencePieceProcessor()
  sp.load(str(model_path))
  return sp


def encode_as_pieces(sp: spm.SentencePieceProcessor, line: str) -> list[str]:
  """Encode a line into SentencePiece tokens (pieces)."""
  return list(sp.encode_as_pieces(line))


def decode_pieces(sp: spm.SentencePieceProcessor, pieces: Sequence[str]) -> str:
  """Decode SentencePiece tokens (pieces) back into text."""
  return sp.decode_pieces(list(pieces))


def get_sp_vocab_list(sp: spm.SentencePieceProcessor) -> list[str]:
  """Return the list of pieces in the SentencePiece vocab, in id order."""
  size = int(sp.get_piece_size())
  return [sp.id_to_piece(i) for i in range(size)]
