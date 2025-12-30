from pathlib import Path
from vocab import VocabEntry, Vocab


def test_vocab_roundtrip(tmp_path: Path):
  src = VocabEntry.build_from_pieces(["a", "b"])
  tgt = VocabEntry.build_from_pieces(["x", "y"])
  v = Vocab(src=src, tgt=tgt)

  p = tmp_path / "vocab.json"
  v.save(p)
  v2 = Vocab.load(p)

  assert v2.src.word2id["<pad>"] == 0
  assert len(v2.src) == len(v.src)
  assert len(v2.tgt) == len(v.tgt)
