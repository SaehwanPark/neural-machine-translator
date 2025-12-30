from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Batch:
  """A padded batch for JAX: arrays are int32, shape (batch, time)."""

  src_ids: np.ndarray
  src_lens: np.ndarray
  tgt_in_ids: np.ndarray
  tgt_out_ids: np.ndarray
  tgt_lens: np.ndarray


def pad_2d(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
  """Pad a list of integer sequences to (batch, max_len) and return (padded, lengths)."""
  lens = np.array([len(s) for s in seqs], dtype=np.int32)
  max_len = int(lens.max()) if len(lens) > 0 else 0
  out = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
  for i, s in enumerate(seqs):
    out[i, : len(s)] = np.asarray(s, dtype=np.int32)
  return out, lens


def batch_iter(
  pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
  batch_size: int,
  shuffle: bool,
  seed: int = 0,
) -> Iterator[list[Tuple[Sequence[int], Sequence[int]]]]:
  """Yield batches sorted by source length (desc), matching the starter pattern. :contentReference[oaicite:5]{index=5}"""
  idx = np.arange(len(pairs))
  if shuffle:
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

  for start in range(0, len(pairs), batch_size):
    chunk = [pairs[i] for i in idx[start : start + batch_size]]
    chunk.sort(key=lambda x: len(x[0]), reverse=True)
    yield chunk


def make_batch(
  chunk: Sequence[Tuple[Sequence[int], Sequence[int]]],
  src_pad_id: int,
  tgt_pad_id: int,
) -> Batch:
  """Create a Batch with teacher forcing: tgt_in (BOS..), tgt_out (..EOS)."""
  src_seqs = [c[0] for c in chunk]
  tgt_seqs = [c[1] for c in chunk]

  src_ids, src_lens = pad_2d(src_seqs, src_pad_id)
  tgt_ids, tgt_lens = pad_2d(tgt_seqs, tgt_pad_id)

  # teacher forcing shift
  tgt_in = tgt_ids[:, :-1]
  tgt_out = tgt_ids[:, 1:]
  return Batch(
    src_ids=src_ids,
    src_lens=src_lens,
    tgt_in_ids=tgt_in,
    tgt_out_ids=tgt_out,
    tgt_lens=tgt_lens,
  )
