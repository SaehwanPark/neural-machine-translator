from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from batching import Batch
from nmt_flax import ModelConfig, NmtFlax


@dataclass(frozen=True)
class TrainConfig:
  """Training hyperparameters."""

  batch_size: int = 32
  lr: float = 1e-3
  clip_norm: float = 5.0
  max_epoch: int = 5
  dropout_rate: float = 0.3
  log_every: int = 200
  seed: int = 0


def cross_entropy_loss(
  logits: jnp.ndarray, targets: jnp.ndarray, pad_id: int
) -> jnp.ndarray:
  """
  Token-average cross entropy with padding mask.

  Args:
    logits: (batch, time, vocab)
    targets: (batch, time)
    pad_id: padding token id in target vocab

  Returns:
    Scalar mean loss over non-pad tokens.
  """
  logp = jax.nn.log_softmax(logits, axis=-1)
  nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)

  mask = (targets != pad_id).astype(jnp.float32)
  denom = jnp.maximum(mask.sum(), 1.0)
  return (nll * mask).sum() / denom


def make_state(
  rng: jax.Array,
  cfg: ModelConfig,
  lr: float,
  clip_norm: float,
) -> train_state.TrainState:
  """
  Initialize params and optimizer state.

  Args:
    rng: PRNG key
    cfg: model config
    lr: learning rate
    clip_norm: global-norm gradient clipping threshold

  Returns:
    Flax TrainState with params + optimizer.
  """
  model = NmtFlax(cfg)

  dummy_src = jnp.zeros((2, 5), dtype=jnp.int32)
  dummy_src_lens = jnp.array([5, 3], dtype=jnp.int32)
  dummy_tgt_in = jnp.zeros((2, 6), dtype=jnp.int32)

  params = model.init(
    {"params": rng},
    dummy_src,
    dummy_src_lens,
    dummy_tgt_in,
    train=True,
    dropout_rng=rng,
  )["params"]

  tx = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adam(lr),
  )
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(
  state: train_state.TrainState,
  batch: Batch,
  pad_id: int,
  dropout_rng: jax.Array,
  model_cfg: ModelConfig,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
  """
  Single optimizer step.

  Args:
    state: TrainState
    batch: batch of ids and lengths
    pad_id: target pad id
    dropout_rng: PRNG key for dropout this step
    model_cfg: model config

  Returns:
    (updated_state, loss)
  """

  def loss_fn(params):
    logits = NmtFlax(model_cfg).apply(
      {"params": params},
      jnp.asarray(batch.src_ids),
      jnp.asarray(batch.src_lens),
      jnp.asarray(batch.tgt_in_ids),
      train=True,
      dropout_rng=dropout_rng,
    )
    return cross_entropy_loss(logits, jnp.asarray(batch.tgt_out_ids), pad_id)

  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


def eval_ppl(
  state: train_state.TrainState,
  batches: list[Batch],
  pad_id: int,
  model_cfg: ModelConfig,
) -> float:
  """
  Compute perplexity.

  Args:
    state: TrainState
    batches: evaluation batches
    pad_id: target pad id
    model_cfg: model config

  Returns:
    Perplexity.
  """
  losses: list[float] = []
  for b in batches:
    logits = NmtFlax(model_cfg).apply(
      {"params": state.params},
      jnp.asarray(b.src_ids),
      jnp.asarray(b.src_lens),
      jnp.asarray(b.tgt_in_ids),
      train=False,
      dropout_rng=None,
    )
    loss = cross_entropy_loss(logits, jnp.asarray(b.tgt_out_ids), pad_id)
    losses.append(float(loss))

  mean_loss = float(np.mean(losses)) if losses else 0.0
  return float(np.exp(mean_loss))
