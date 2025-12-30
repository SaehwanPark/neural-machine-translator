from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class ModelConfig:
  """Model hyperparameters."""

  src_vocab_size: int
  tgt_vocab_size: int
  embed_size: int = 256
  hidden_size: int = 256
  dropout_rate: float = 0.2


def _mask_from_lens(lens: jnp.ndarray, max_len: int) -> jnp.ndarray:
  """Return boolean mask (batch, max_len): True for valid positions."""
  idx = jnp.arange(max_len)[None, :]
  return idx < lens[:, None]


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
  """Numerically stable sigmoid."""
  return jax.nn.sigmoid(x)


def _dropout(x: jnp.ndarray, rng: jax.Array, rate: float, train: bool) -> jnp.ndarray:
  """
  Pure dropout (no Linen module calls).

  Args:
    x: input
    rng: PRNG key
    rate: dropout rate
    train: whether to apply dropout

  Returns:
    x after dropout (inverted dropout scaling).
  """
  if (not train) or rate <= 0.0:
    return x
  keep_prob = 1.0 - rate
  mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
  return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


def _lstm_step(
  w_x: jnp.ndarray,
  w_h: jnp.ndarray,
  b: jnp.ndarray,
  carry: Tuple[jnp.ndarray, jnp.ndarray],
  x_t: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """
  One LSTM step implemented as pure JAX.

  Flax convention: carry is (c, h).

  Shapes:
    x_t: (batch, in_dim)
    carry: (c, h) each (batch, hidden)
    w_x: (in_dim, 4*hidden)
    w_h: (hidden, 4*hidden)
    b: (4*hidden,)

  Returns:
    new_carry: (c, h)
    h_t: (batch, hidden)
  """
  c, h = carry
  gates = jnp.dot(x_t, w_x) + jnp.dot(h, w_h) + b
  i, f, g, o = jnp.split(gates, 4, axis=-1)

  i = _sigmoid(i)
  f = _sigmoid(f)
  o = _sigmoid(o)
  g = jnp.tanh(g)

  new_c = f * c + i * g
  new_h = o * jnp.tanh(new_c)
  return (new_c, new_h), new_h


def _apply_lstm_scan(
  w_x: jnp.ndarray,
  w_h: jnp.ndarray,
  b: jnp.ndarray,
  xs: jnp.ndarray,
  lens: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """
  Run an LSTM over time with padding-aware masking using lax.scan.

  Args:
    xs: (batch, time, in_dim)
    lens: (batch,)

  Returns:
    final_carry: (c_T, h_T) each (batch, hidden)
    hs: (batch, time, hidden)
  """
  batch, time, _ = xs.shape
  hidden = int(b.shape[0] // 4)

  init = (
    jnp.zeros((batch, hidden), dtype=xs.dtype),  # c
    jnp.zeros((batch, hidden), dtype=xs.dtype),  # h
  )

  mask = _mask_from_lens(lens, time)  # (batch, time)

  def step(carry, inp):
    x_t, m_t = inp  # x_t: (batch, in_dim), m_t: (batch,)
    new_carry, h_t = _lstm_step(w_x, w_h, b, carry, x_t)

    m = m_t[:, None]
    carry_out = jax.tree_util.tree_map(
      lambda n, o: jnp.where(m, n, o), new_carry, carry
    )
    h_out = jnp.where(m, h_t, jnp.zeros_like(h_t))
    return carry_out, h_out

  final_carry, hs_t = jax.lax.scan(
    step,
    init,
    (xs.swapaxes(0, 1), mask.swapaxes(0, 1)),
  )
  hs = hs_t.swapaxes(0, 1)
  return final_carry, hs


class NmtFlax(nn.Module):
  """
  biLSTM encoder + dot-product attention + LSTM decoder with input feeding.

  Important implementation detail (Flax 0.12.2 / JAX 0.8.2):
    - No Linen submodule calls inside jax.lax.scan.
    - All scan computations are pure JAX math with explicit parameters.
  """

  cfg: ModelConfig

  def setup(self) -> None:
    self.src_embed = nn.Embed(self.cfg.src_vocab_size, self.cfg.embed_size)
    self.tgt_embed = nn.Embed(self.cfg.tgt_vocab_size, self.cfg.embed_size)

    self.post_embed_cnn = nn.Conv(
      features=self.cfg.embed_size,
      kernel_size=(2,),
      padding="SAME",
      use_bias=True,
    )

    # These Dense layers are called OUTSIDE scan.
    self.h_proj = nn.Dense(self.cfg.hidden_size, use_bias=False)
    self.c_proj = nn.Dense(self.cfg.hidden_size, use_bias=False)
    self.att_proj = nn.Dense(self.cfg.hidden_size, use_bias=False)

    # LSTM params (pure) for encoder fwd/bwd and decoder.
    h = self.cfg.hidden_size
    e = self.cfg.embed_size

    kx = nn.initializers.xavier_uniform()
    kh = nn.initializers.orthogonal()
    kb = nn.initializers.zeros

    # Encoder forward: input dim = embed_size
    self.enc_fwd_wx = self.param("enc_fwd_wx", kx, (e, 4 * h))
    self.enc_fwd_wh = self.param("enc_fwd_wh", kh, (h, 4 * h))
    self.enc_fwd_b = self.param("enc_fwd_b", kb, (4 * h,))

    # Encoder backward: input dim = embed_size
    self.enc_bwd_wx = self.param("enc_bwd_wx", kx, (e, 4 * h))
    self.enc_bwd_wh = self.param("enc_bwd_wh", kh, (h, 4 * h))
    self.enc_bwd_b = self.param("enc_bwd_b", kb, (4 * h,))

    # Decoder LSTM: input dim = embed_size + hidden_size (input feeding)
    dec_in = e + h
    self.dec_wx = self.param("dec_wx", kx, (dec_in, 4 * h))
    self.dec_wh = self.param("dec_wh", kh, (h, 4 * h))
    self.dec_b = self.param("dec_b", kb, (4 * h,))

    # Combined projection (Luong input feeding) and vocab projection.
    # u_t = [dec_h; a_t] where a_t is (2h) -> u_t dim = 3h
    self.comb_w = self.param("comb_w", kx, (3 * h, h))
    self.comb_b = self.param("comb_b", kb, (h,))

    self.vocab_w = self.param("vocab_w", kx, (h, self.cfg.tgt_vocab_size))
    self.vocab_b = self.param("vocab_b", kb, (self.cfg.tgt_vocab_size,))

  def encode(
    self,
    src_ids: jnp.ndarray,
    src_lens: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Encode source.

    Args:
      src_ids: (batch, src_len)
      src_lens: (batch,)

    Returns:
      enc_hiddens: (batch, src_len, 2h)
      dec_init: (c0, h0) each (batch, h)
    """
    x = self.src_embed(src_ids)  # (b, t, e)
    x = self.post_embed_cnn(x)  # (b, t, e)

    (c_f, h_f), hs_f = _apply_lstm_scan(
      self.enc_fwd_wx, self.enc_fwd_wh, self.enc_fwd_b, x, src_lens
    )

    x_rev = jnp.flip(x, axis=1)
    (c_b, h_b), hs_b_rev = _apply_lstm_scan(
      self.enc_bwd_wx, self.enc_bwd_wh, self.enc_bwd_b, x_rev, src_lens
    )
    hs_b = jnp.flip(hs_b_rev, axis=1)

    enc = jnp.concatenate([hs_f, hs_b], axis=-1)  # (b, t, 2h)

    h_cat = jnp.concatenate([h_f, h_b], axis=-1)  # (b, 2h)
    c_cat = jnp.concatenate([c_f, c_b], axis=-1)  # (b, 2h)

    h0 = self.h_proj(h_cat)
    c0 = self.c_proj(c_cat)
    return enc, (c0, h0)

  def _attend(
    self,
    enc: jnp.ndarray,
    enc_proj: jnp.ndarray,
    src_lens: jnp.ndarray,
    dec_h: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Dot-product attention.

    Args:
      enc: (b, src_len, 2h)
      enc_proj: (b, src_len, h)
      src_lens: (b,)
      dec_h: (b, h)

    Returns:
      a_t: (b, 2h)
      alpha: (b, src_len)
    """
    scores = jnp.einsum("bth,bh->bt", enc_proj, dec_h)  # (b, t)
    b, t, _ = enc.shape
    valid = _mask_from_lens(src_lens, t)
    scores = jnp.where(valid, scores, -1e9)

    alpha = jax.nn.softmax(scores, axis=-1)
    a_t = jnp.einsum("bt,bth->bh", alpha, enc)  # (b, 2h)
    return a_t, alpha

  def decode(
    self,
    enc: jnp.ndarray,
    src_lens: jnp.ndarray,
    dec_init: Tuple[jnp.ndarray, jnp.ndarray],
    tgt_in_ids: jnp.ndarray,
    *,
    train: bool,
    dropout_rng: jax.Array | None,
  ) -> jnp.ndarray:
    """
    Teacher-forced decoding with input feeding.

    Args:
      enc: (b, src_len, 2h)
      src_lens: (b,)
      dec_init: (c0, h0) each (b, h)
      tgt_in_ids: (b, tgt_len) input tokens starting with <s>
      train: whether to apply dropout
      dropout_rng: RNG for dropout (required if train=True and dropout_rate>0)

    Returns:
      logits: (b, tgt_len, vocab)
    """
    enc_proj = self.att_proj(enc)  # (b, src_len, h)
    y = self.tgt_embed(tgt_in_ids)  # (b, t, e)

    b, t, _ = y.shape
    carry = dec_init  # (c, h)
    o_prev = jnp.zeros((b, self.cfg.hidden_size), dtype=y.dtype)

    # Provide keys per timestep (stable scan signature).
    if dropout_rng is None:
      dropout_rng = jax.random.PRNGKey(0)
    keys = jax.random.split(dropout_rng, t)

    def step(state, inp):
      carry_local, o_prev_local = state
      y_t, k_t = inp

      x_t = jnp.concatenate([y_t, o_prev_local], axis=-1)  # (b, e+h)
      new_carry, dec_h = _lstm_step(
        self.dec_wx, self.dec_wh, self.dec_b, carry_local, x_t
      )

      a_t, _ = self._attend(enc, enc_proj, src_lens, dec_h)  # (b, 2h)

      u_t = jnp.concatenate([dec_h, a_t], axis=-1)  # (b, 3h)
      v_t = jnp.dot(u_t, self.comb_w) + self.comb_b  # (b, h)
      o_t = jnp.tanh(v_t)
      o_t = _dropout(o_t, k_t, self.cfg.dropout_rate, train)

      logits_t = jnp.dot(o_t, self.vocab_w) + self.vocab_b  # (b, V)
      return (new_carry, o_t), logits_t

    (_, _), logits_t = jax.lax.scan(
      step,
      (carry, o_prev),
      (y.swapaxes(0, 1), keys),
    )
    logits = logits_t.swapaxes(0, 1)  # (b, t, V)
    return logits

  def __call__(
    self,
    src_ids: jnp.ndarray,
    src_lens: jnp.ndarray,
    tgt_in_ids: jnp.ndarray,
    *,
    train: bool,
    dropout_rng: jax.Array | None = None,
  ) -> jnp.ndarray:
    """Forward pass returning logits for teacher forcing."""
    enc, dec_init = self.encode(src_ids, src_lens)
    return self.decode(
      enc, src_lens, dec_init, tgt_in_ids, train=train, dropout_rng=dropout_rng
    )

  def decode_step(
    self,
    enc: jnp.ndarray,
    enc_proj: jnp.ndarray,
    src_lens: jnp.ndarray,
    carry: Tuple[jnp.ndarray, jnp.ndarray],
    o_prev: jnp.ndarray,
    y_id: jnp.ndarray,
    *,
    train: bool,
    dropout_key: jax.Array | None,
  ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Single decoder step (useful for beam search).

    Args:
      enc: (b, src_len, 2h)
      enc_proj: (b, src_len, h)
      src_lens: (b,)
      carry: (c, h) each (b, h)
      o_prev: (b, h)
      y_id: (b,) int token ids
      train: whether to apply dropout
      dropout_key: RNG for dropout

    Returns:
      new_carry: (c, h)
      new_o: (b, h)
      logits: (b, V)
    """
    y_t = self.tgt_embed(y_id)  # (b, e)
    x_t = jnp.concatenate([y_t, o_prev], axis=-1)  # (b, e+h)

    new_carry, dec_h = _lstm_step(self.dec_wx, self.dec_wh, self.dec_b, carry, x_t)
    a_t, _ = self._attend(enc, enc_proj, src_lens, dec_h)

    u_t = jnp.concatenate([dec_h, a_t], axis=-1)
    v_t = jnp.dot(u_t, self.comb_w) + self.comb_b
    o_t = jnp.tanh(v_t)

    if dropout_key is None:
      dropout_key = jax.random.PRNGKey(0)
    o_t = _dropout(o_t, dropout_key, self.cfg.dropout_rate, train)

    logits = jnp.dot(o_t, self.vocab_w) + self.vocab_b
    return new_carry, o_t, logits
