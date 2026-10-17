"""A vanilla tanh RNN exposing token-sequence -> next-token logits."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


class RNN(eqx.Module):
    embedding: eqx.nn.Embedding
    input_projection: eqx.nn.Linear
    recurrent_projection: eqx.nn.Linear
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self, vocab_size: int, hidden_size: int = 128, *, key: PRNGKeyArray
    ):
        embedding_key, input_key, recurrent_key, output_key = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=embedding_key)
        self.input_projection = eqx.nn.Linear(hidden_size, hidden_size, key=input_key)
        self.recurrent_projection = eqx.nn.Linear(
            hidden_size, hidden_size, use_bias=False, key=recurrent_key
        )
        self.readout = eqx.nn.Linear(hidden_size, vocab_size, key=output_key)

    def __call__(self, tokens: Int[Array, "time"]) -> Float[Array, "time vocab"]:
        inputs = jax.vmap(self.embedding)(tokens)

        def step(hidden, inputs):
            hidden = jnp.tanh(
                self.input_projection(inputs) + self.recurrent_projection(hidden)
            )
            return hidden, self.readout(hidden)

        _, logits = jax.lax.scan(
            step, jnp.zeros(self.hidden_size, dtype=inputs.dtype), inputs
        )
        return logits
