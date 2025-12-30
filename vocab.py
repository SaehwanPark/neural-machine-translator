from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<s>", "</s>", "<unk>")


@dataclass(frozen=True)
class VocabEntry:
  """A single vocabulary mapping tokens <-> ids."""

  word2id: Dict[str, int]
  id2word: Dict[int, str]
  unk_id: int
  pad_id: int
  bos_id: int
  eos_id: int

  @staticmethod
  def build_from_pieces(pieces: Iterable[str]) -> "VocabEntry":
    """Build a vocab entry from SentencePiece pieces plus special tokens."""
    word2id: Dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
      word2id[tok] = len(word2id)

    for p in pieces:
      if p not in word2id:
        word2id[p] = len(word2id)

    id2word = {i: w for w, i in word2id.items()}
    return VocabEntry(
      word2id=word2id,
      id2word=id2word,
      unk_id=word2id["<unk>"],
      pad_id=word2id["<pad>"],
      bos_id=word2id["<s>"],
      eos_id=word2id["</s>"],
    )

  def __len__(self) -> int:
    return len(self.word2id)

  def token_to_id(self, tok: str) -> int:
    return self.word2id.get(tok, self.unk_id)

  def ids(self, toks: Sequence[str]) -> List[int]:
    return [self.token_to_id(t) for t in toks]

  def tokens(self, ids: Sequence[int]) -> List[str]:
    return [self.id2word[int(i)] for i in ids]


@dataclass(frozen=True)
class Vocab:
  """Source + target vocab."""

  src: VocabEntry
  tgt: VocabEntry

  def save(self, path: Path) -> None:
    """Save vocab as JSON."""
    payload = {"src_word2id": self.src.word2id, "tgt_word2id": self.tgt.word2id}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

  @staticmethod
  def load(path: Path) -> "Vocab":
    """Load vocab from JSON."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    src_w2i = {str(k): int(v) for k, v in obj["src_word2id"].items()}
    tgt_w2i = {str(k): int(v) for k, v in obj["tgt_word2id"].items()}

    def entry_from_word2id(w2i: Dict[str, int]) -> VocabEntry:
      id2w = {i: w for w, i in w2i.items()}
      return VocabEntry(
        word2id=w2i,
        id2word=id2w,
        unk_id=w2i["<unk>"],
        pad_id=w2i["<pad>"],
        bos_id=w2i["<s>"],
        eos_id=w2i["</s>"],
      )

    return Vocab(src=entry_from_word2id(src_w2i), tgt=entry_from_word2id(tgt_w2i))
