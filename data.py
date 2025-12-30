from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from paths import get_data_dir
from tokenization import (
  SpmConfig,
  train_sentencepiece,
  load_sp,
  encode_as_pieces,
  get_sp_vocab_list,
)
from vocab import Vocab, VocabEntry


@dataclass(frozen=True)
class DataPaths:
  """Paths for train/dev/test parallel corpora."""

  train_src: Path
  train_tgt: Path
  dev_src: Path
  dev_tgt: Path
  test_src: Path
  test_tgt: Path


def get_data_paths(
  src_lang: str, tgt_lang: str, data_dir: Path | None = None
) -> DataPaths:
  """
  Resolve corpus file paths under DATA_DIR using language abbreviations as extensions.

  Example:
    train.zh, train.en, dev.zh, dev.en, test.zh, test.en
  """
  d = (data_dir or get_data_dir()).resolve()
  return DataPaths(
    train_src=d / f"train.{src_lang}",
    train_tgt=d / f"train.{tgt_lang}",
    dev_src=d / f"dev.{src_lang}",
    dev_tgt=d / f"dev.{tgt_lang}",
    test_src=d / f"test.{src_lang}",
    test_tgt=d / f"test.{tgt_lang}",
  )


def get_artifact_dir(base_out_dir: Path, src_lang: str, tgt_lang: str) -> Path:
  """Return an artifact directory scoped to a language pair, e.g. artifacts/zh-en."""
  return (base_out_dir / f"{src_lang}-{tgt_lang}").resolve()


def read_lines(path: Path) -> List[str]:
  """Read a UTF-8 text file as a list of lines."""
  return path.read_text(encoding="utf-8").splitlines()


def write_meta_json(
  out_dir: Path,
  src_lang: str,
  tgt_lang: str,
  src_vocab_size: int,
  tgt_vocab_size: int,
) -> Path:
  """Write artifacts metadata to out_dir/meta.json and return the path."""
  payload = {
    "src_lang": src_lang,
    "tgt_lang": tgt_lang,
    "src_vocab_size": int(src_vocab_size),
    "tgt_vocab_size": int(tgt_vocab_size),
    "created_utc": datetime.now(timezone.utc).isoformat(),
  }
  out_path = out_dir / "meta.json"
  out_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  return out_path


def prepare_tokenizers_and_vocab(
  out_dir: Path,
  train_src: Path,
  train_tgt: Path,
  src_lang: str,
  tgt_lang: str,
  src_vocab_size: int = 21000,
  tgt_vocab_size: int = 8000,
) -> tuple[Path, Path, Vocab]:
  """
  Train/load SentencePiece models and build vocab JSON.

  Artifacts:
    - spm.<src_lang>.model
    - spm.<tgt_lang>.model
    - vocab.json
    - meta.json
  """
  out_dir.mkdir(parents=True, exist_ok=True)

  src_model = out_dir / f"spm.{src_lang}.model"
  tgt_model = out_dir / f"spm.{tgt_lang}.model"

  if not src_model.exists():
    trained = train_sentencepiece(
      train_src,
      out_dir,
      SpmConfig(vocab_size=src_vocab_size, model_prefix=f"spm.{src_lang}"),
    )
    src_model = trained

  if not tgt_model.exists():
    trained = train_sentencepiece(
      train_tgt,
      out_dir,
      SpmConfig(vocab_size=tgt_vocab_size, model_prefix=f"spm.{tgt_lang}"),
    )
    tgt_model = trained

  src_sp = load_sp(src_model)
  tgt_sp = load_sp(tgt_model)

  src_vocab = VocabEntry.build_from_pieces(get_sp_vocab_list(src_sp))
  tgt_vocab = VocabEntry.build_from_pieces(get_sp_vocab_list(tgt_sp))

  write_meta_json(out_dir, src_lang, tgt_lang, src_vocab_size, tgt_vocab_size)
  return src_model, tgt_model, Vocab(src=src_vocab, tgt=tgt_vocab)


def build_parallel_ids(
  src_model: Path,
  tgt_model: Path,
  vocab: Vocab,
  src_lines: List[str],
  tgt_lines: List[str],
) -> List[Tuple[List[int], List[int]]]:
  """
  Tokenize and map to ids.

  Target sequences are wrapped with <s> and </s> for teacher forcing.
  """
  src_sp = load_sp(src_model)
  tgt_sp = load_sp(tgt_model)

  pairs: List[Tuple[List[int], List[int]]] = []
  for s, t in zip(src_lines, tgt_lines):
    s_pieces = encode_as_pieces(src_sp, s)
    t_pieces = ["<s>"] + encode_as_pieces(tgt_sp, t) + ["</s>"]
    pairs.append((vocab.src.ids(s_pieces), vocab.tgt.ids(t_pieces)))
  return pairs
