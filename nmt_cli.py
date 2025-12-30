from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax
from flax.training import checkpoints
from tqdm import tqdm

from batching import batch_iter, make_batch
from data import (
  get_data_paths,
  get_artifact_dir,
  prepare_tokenizers_and_vocab,
  build_parallel_ids,
  read_lines,
)
from logging_utils import LoggingConfig, configure_logging
from nmt_flax import ModelConfig
from train_nmt import TrainConfig, make_state, train_step, eval_ppl
from vocab import Vocab
from decode_nmt import beam_search_one
from tokenization import load_sp


logger = logging.getLogger(__name__)


def _artifact_paths(
  base_out_dir: Path,
  src_lang: str,
  tgt_lang: str,
) -> tuple[Path, Path, Path, Path]:
  """Return (pair_dir, src_model, tgt_model, vocab_path)."""
  pair_dir = get_artifact_dir(base_out_dir, src_lang, tgt_lang)
  src_model = pair_dir / f"spm.{src_lang}.model"
  tgt_model = pair_dir / f"spm.{tgt_lang}.model"
  vocab_path = pair_dir / "vocab.json"
  return pair_dir, src_model, tgt_model, vocab_path


def cmd_prepare(args: argparse.Namespace) -> None:
  paths = get_data_paths(args.src_lang, args.tgt_lang)
  base_out_dir = Path(args.out_dir)
  pair_dir = get_artifact_dir(base_out_dir, args.src_lang, args.tgt_lang)

  src_model, tgt_model, vocab = prepare_tokenizers_and_vocab(
    out_dir=pair_dir,
    train_src=paths.train_src,
    train_tgt=paths.train_tgt,
    src_lang=args.src_lang,
    tgt_lang=args.tgt_lang,
    src_vocab_size=args.src_vocab_size,
    tgt_vocab_size=args.tgt_vocab_size,
  )

  vocab_path = pair_dir / "vocab.json"
  vocab.save(vocab_path)

  logger.info("prepared artifacts_dir=%s", pair_dir)
  logger.info("saved: %s", src_model)
  logger.info("saved: %s", tgt_model)
  logger.info("saved: %s", vocab_path)
  logger.info("saved: %s", pair_dir / "meta.json")


def cmd_train(args: argparse.Namespace) -> None:
  paths = get_data_paths(args.src_lang, args.tgt_lang)
  base_out_dir = Path(args.out_dir)
  pair_dir, src_model, tgt_model, vocab_path = _artifact_paths(
    base_out_dir, args.src_lang, args.tgt_lang
  )

  vocab = Vocab.load(vocab_path)

  train_pairs = build_parallel_ids(
    src_model,
    tgt_model,
    vocab,
    read_lines(paths.train_src),
    read_lines(paths.train_tgt),
  )
  dev_pairs = build_parallel_ids(
    src_model,
    tgt_model,
    vocab,
    read_lines(paths.dev_src),
    read_lines(paths.dev_tgt),
  )

  model_cfg = ModelConfig(
    src_vocab_size=len(vocab.src),
    tgt_vocab_size=len(vocab.tgt),
    embed_size=args.embed_size,
    hidden_size=args.hidden_size,
    dropout_rate=args.dropout,
  )
  train_cfg = TrainConfig(
    batch_size=args.batch_size,
    lr=args.lr,
    clip_norm=args.clip_norm,
    max_epoch=args.max_epoch,
    dropout_rate=args.dropout,
    seed=args.seed,
  )

  rng = jax.random.PRNGKey(train_cfg.seed)
  state = make_state(rng, model_cfg, train_cfg.lr, train_cfg.clip_norm)

  step = 0
  ckpt_dir = pair_dir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  logger.info(
    "starting training pair=%s-%s artifacts_dir=%s",
    args.src_lang,
    args.tgt_lang,
    pair_dir,
  )

  for epoch in range(train_cfg.max_epoch):
    for chunk in batch_iter(
      train_pairs,
      train_cfg.batch_size,
      shuffle=True,
      seed=train_cfg.seed + epoch,
    ):
      batch = make_batch(chunk, vocab.src.pad_id, vocab.tgt.pad_id)
      rng, dr = jax.random.split(rng)
      state, loss = train_step(state, batch, vocab.tgt.pad_id, dr, model_cfg)
      step += 1

      if step % args.log_every == 0:
        dev_batches = []
        for c in batch_iter(
          dev_pairs[: args.dev_eval_examples],
          train_cfg.batch_size,
          shuffle=False,
        ):
          dev_batches.append(make_batch(c, vocab.src.pad_id, vocab.tgt.pad_id))
        ppl = eval_ppl(state, dev_batches, vocab.tgt.pad_id, model_cfg)
        logger.info(
          "epoch=%d step=%d loss=%.4f dev_ppl=%.2f",
          epoch + 1,
          step,
          float(loss),
          ppl,
        )
        checkpoints.save_checkpoint(
          str(ckpt_dir),
          state.params,
          step,
          keep=3,
          overwrite=True,
        )

  checkpoints.save_checkpoint(str(ckpt_dir), state.params, step, keep=3, overwrite=True)
  logger.info("done. last_step=%d ckpt_dir=%s", step, ckpt_dir)


def cmd_decode(args: argparse.Namespace) -> None:
  base_out_dir = Path(args.out_dir)
  pair_dir, src_model, tgt_model, vocab_path = _artifact_paths(
    base_out_dir, args.src_lang, args.tgt_lang
  )

  vocab = Vocab.load(vocab_path)

  ckpt_dir = pair_dir / "checkpoints"
  params = checkpoints.restore_checkpoint(str(ckpt_dir), target=None)
  if params is None:
    raise RuntimeError(f"no checkpoint found in: {ckpt_dir}")

  model_cfg = ModelConfig(
    src_vocab_size=len(vocab.src),
    tgt_vocab_size=len(vocab.tgt),
    embed_size=args.embed_size,
    hidden_size=args.hidden_size,
    dropout_rate=0.0,
  )

  src_sp = load_sp(src_model)
  tgt_sp = load_sp(tgt_model)

  src_lines = read_lines(Path(args.src_file))
  out_lines: list[str] = []

  logger.info(
    "decoding pair=%s-%s src_file=%s out_file=%s",
    args.src_lang,
    args.tgt_lang,
    args.src_file,
    args.out_file,
  )

  for line in tqdm(src_lines, desc="Decoding"):
    pieces = src_sp.encode_as_pieces(line)
    src_ids = vocab.src.ids(list(pieces))

    hyps = beam_search_one(
      params,
      model_cfg,
      vocab,
      src_ids,
      max_len=args.max_len,
      beam_size=args.beam_size,
    )
    best_ids = hyps[0].token_ids

    best_pieces = vocab.tgt.tokens(best_ids)
    best_pieces = [p for p in best_pieces if p not in ("<s>", "</s>", "<pad>")]
    sent = tgt_sp.decode_pieces(best_pieces)
    out_lines.append(sent)

  out_path = Path(args.out_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
  logger.info("wrote: %s", out_path)


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser()

  p.add_argument("--log-level", type=str, default="INFO")
  p.add_argument("--log-file", type=str, default="")
  p.add_argument("--log-use-tqdm-handler", action="store_true")

  p.add_argument("--src-lang", type=str, default="zh")
  p.add_argument("--tgt-lang", type=str, default="en")

  sub = p.add_subparsers(dest="cmd", required=True)

  sp = sub.add_parser("prepare")
  sp.add_argument("--out-dir", type=str, default="artifacts")
  sp.add_argument("--src-vocab-size", type=int, default=21000)
  sp.add_argument("--tgt-vocab-size", type=int, default=8000)

  st = sub.add_parser("train")
  st.add_argument("--out-dir", type=str, default="artifacts")
  st.add_argument("--seed", type=int, default=0)
  st.add_argument("--batch-size", type=int, default=32)
  st.add_argument("--embed-size", type=int, default=256)
  st.add_argument("--hidden-size", type=int, default=256)
  st.add_argument("--dropout", type=float, default=0.3)
  st.add_argument("--lr", type=float, default=1e-3)
  st.add_argument("--clip-norm", type=float, default=5.0)
  st.add_argument("--max-epoch", type=int, default=5)
  st.add_argument("--log-every", type=int, default=200)
  st.add_argument("--dev-eval-examples", type=int, default=1024)

  sd = sub.add_parser("decode")
  sd.add_argument("--out-dir", type=str, default="artifacts")
  sd.add_argument("--embed-size", type=int, default=256)
  sd.add_argument("--hidden-size", type=int, default=256)
  sd.add_argument("--beam-size", type=int, default=5)
  sd.add_argument("--max-len", type=int, default=70)
  sd.add_argument("--src-file", type=str, required=True)
  sd.add_argument("--out-file", type=str, required=True)

  return p


def main() -> None:
  args = build_argparser().parse_args()

  log_file = Path(args.log_file) if args.log_file else None
  configure_logging(
    LoggingConfig(
      level=args.log_level,
      log_file=log_file,
      use_tqdm_handler=bool(args.log_use_tqdm_handler),
    )
  )

  logger.info(
    "command=%s src_lang=%s tgt_lang=%s", args.cmd, args.src_lang, args.tgt_lang
  )

  if args.cmd == "prepare":
    cmd_prepare(args)
  elif args.cmd == "train":
    cmd_train(args)
  elif args.cmd == "decode":
    cmd_decode(args)
  else:
    raise RuntimeError(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
  main()
