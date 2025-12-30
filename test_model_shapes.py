import jax
import jax.numpy as jnp

from nmt_flax import NmtFlax, ModelConfig


def test_forward_logits_shape():
  cfg = ModelConfig(
    src_vocab_size=100,
    tgt_vocab_size=120,
    embed_size=16,
    hidden_size=8,
    dropout_rate=0.0,
  )
  model = NmtFlax(cfg)

  rng = jax.random.PRNGKey(0)
  src = jnp.zeros((4, 7), dtype=jnp.int32)
  src_lens = jnp.array([7, 6, 4, 2], dtype=jnp.int32)
  tgt_in = jnp.zeros((4, 5), dtype=jnp.int32)

  vars = model.init({"params": rng}, src, src_lens, tgt_in, train=False)
  logits = model.apply(vars, src, src_lens, tgt_in, train=False)
  assert logits.shape == (4, 5, 120)
