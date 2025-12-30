from __future__ import annotations

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp

from nmt_flax import ModelConfig, NmtFlax
from vocab import Vocab


@dataclass(frozen=True)
class Hypothesis:
  """Beam search hypothesis."""

  token_ids: List[int]
  score: float


def beam_search_one(
  params,
  model_cfg: ModelConfig,
  vocab: Vocab,
  src_ids: List[int],
  *,
  max_len: int = 70,
  beam_size: int = 5,
) -> List[Hypothesis]:
  """
  Beam search for a single source sentence.

  Notes:
    - Decoder carry follows Flax convention: (c, h).
    - Uses NmtFlax.encode and NmtFlax.decode_step.

  Args:
    params: model params pytree
    model_cfg: model config
    vocab: vocabulary (must provide bos/eos ids on vocab.tgt)
    src_ids: token ids (already SentencePiece’d and vocab-mapped)
    max_len: max target length
    beam_size: beam size

  Returns:
    Top hypotheses sorted by score desc.
  """
  model = NmtFlax(model_cfg)

  src = jnp.asarray([src_ids], dtype=jnp.int32)
  src_lens = jnp.asarray([len(src_ids)], dtype=jnp.int32)

  enc, dec_init = model.apply({"params": params}, src, src_lens, method=model.encode)
  enc_proj = model.apply({"params": params}, enc, method=lambda m, e: m.att_proj(e))

  c0, h0 = dec_init
  o0 = jnp.zeros((1, model_cfg.hidden_size), dtype=jnp.float32)

  live: List[Hypothesis] = [Hypothesis([vocab.tgt.bos_id], 0.0)]
  live_c = c0
  live_h = h0
  live_o = o0
  completed: List[Hypothesis] = []

  for _t in range(max_len):
    if len(live) == 0 or len(completed) >= beam_size:
      break

    live = sorted(live, key=lambda x: x.score, reverse=True)[:beam_size]
    L = len(live)

    live_c = live_c[:L]
    live_h = live_h[:L]
    live_o = live_o[:L]

    last_ids = jnp.asarray([h.token_ids[-1] for h in live], dtype=jnp.int32)
    base_scores = jnp.asarray([h.score for h in live], dtype=jnp.float32)

    enc_L = jnp.tile(enc, (L, 1, 1))
    enc_proj_L = jnp.tile(enc_proj, (L, 1, 1))
    lens_L = jnp.tile(src_lens, (L,))

    new_carry, new_o, logits = model.apply(
      {"params": params},
      enc_L,
      enc_proj_L,
      lens_L,
      (live_c, live_h),
      live_o,
      last_ids,
      train=False,
      dropout_key=None,
      method=model.decode_step,
    )
    new_c, new_h = new_carry

    logp = jax.nn.log_softmax(logits, axis=-1)  # (L, V)
    V = logp.shape[-1]

    cand_scores = (base_scores[:, None] + logp).reshape(-1)
    need = max(beam_size - len(completed), 1)
    topk = jnp.argsort(-cand_scores)[:need]

    next_live: List[Hypothesis] = []
    next_c: List[jnp.ndarray] = []
    next_h: List[jnp.ndarray] = []
    next_o: List[jnp.ndarray] = []

    for flat_i in list(map(int, topk)):
      prev = flat_i // V
      wid = int(flat_i % V)
      sc = float(cand_scores[flat_i])

      seq = live[prev].token_ids + [wid]
      if wid == vocab.tgt.eos_id:
        completed.append(Hypothesis(seq, sc))
        if len(completed) >= beam_size:
          break
      else:
        next_live.append(Hypothesis(seq, sc))
        next_c.append(new_c[prev])
        next_h.append(new_h[prev])
        next_o.append(new_o[prev])

    live = next_live
    if len(live) == 0:
      break

    live_c = jnp.stack(next_c, axis=0)
    live_h = jnp.stack(next_h, axis=0)
    live_o = jnp.stack(next_o, axis=0)

  if not completed:
    completed = live
  return sorted(completed, key=lambda x: x.score, reverse=True)[:beam_size]
