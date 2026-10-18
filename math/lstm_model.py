"""A single-layer LSTM exposing token-sequence -> causal next-token logits."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from data import VOCAB


class LSTM(eqx.Module):
    embedding: eqx.nn.Embedding
    input_projection: eqx.nn.Linear
    recurrent_projection: eqx.nn.Linear
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    forget_bias: float = eqx.field(static=True)

    def __init__(
        self, vocab_size: int, hidden_size: int = 128, *,
        embedding_size: int | None = None, forget_bias: float = 1.0,
        key: PRNGKeyArray,
    ):
        embedding_key, input_key, recurrent_key, output_key = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.forget_bias = forget_bias
        self.embedding = eqx.nn.Embedding(vocab_size, self.embedding_size, key=embedding_key)
        self.input_projection = eqx.nn.Linear(
            self.embedding_size, 4 * hidden_size, key=input_key,
        )
        self.recurrent_projection = eqx.nn.Linear(
            hidden_size, 4 * hidden_size, use_bias=False, key=recurrent_key,
        )
        # Gate order is input, forget, candidate, output. This is an initialization
        # value only; the forget bias remains trainable as part of the projection.
        bias = self.input_projection.bias.at[hidden_size:2 * hidden_size].set(forget_bias)
        self.input_projection = eqx.tree_at(
            lambda layer: layer.bias, self.input_projection, bias,
        )
        self.readout = eqx.nn.Linear(hidden_size, vocab_size, key=output_key)

    def save_artifact(self, config: dict, output_dir: str | Path) -> Path:
        """Save under the Trackio run name, including all reconstruction settings."""
        run_name = config["name"]
        if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("Artifact run name must be a single directory name")
        directory = Path(output_dir) / run_name
        directory.mkdir(parents=True)
        config = {
            **config,
            "architecture": type(self).__name__,
            "model_config": {
                "hidden_size": self.hidden_size,
                "embedding_size": self.embedding_size,
                "forget_bias": self.forget_bias,
            },
            "vocab": VOCAB,
        }
        eqx.tree_serialise_leaves(directory / "model.eqx", self)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return directory

    @classmethod
    def load_artifact(cls, directory: str | Path) -> tuple["LSTM", dict]:
        """Return (model, config) from saved LSTM inference weights and metadata."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config["vocab"] != VOCAB:
            raise ValueError("Artifact vocabulary does not match the current vocabulary")
        if config["architecture"] != cls.__name__:
            raise ValueError("Artifact architecture does not match this model")
        model = eqx.filter_eval_shape(
            cls, len(VOCAB), **config["model_config"], key=jax.random.key(0),
        )
        return eqx.tree_deserialise_leaves(directory / "model.eqx", model), config

    def __call__(self, tokens: Int[Array, "time"]) -> Float[Array, "time vocab"]:
        """Return causal logits, resetting hidden and cell state for each equation."""
        inputs = jax.vmap(self.embedding)(tokens)

        def step(state, inputs):
            hidden, cell = state
            gates = self.input_projection(inputs) + self.recurrent_projection(hidden)
            input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4)
            cell = (
                jax.nn.sigmoid(forget_gate) * cell
                + jax.nn.sigmoid(input_gate) * jnp.tanh(candidate)
            )
            hidden = jax.nn.sigmoid(output_gate) * jnp.tanh(cell)
            return (hidden, cell), self.readout(hidden)

        zeros = jnp.zeros(self.hidden_size, dtype=inputs.dtype)
        _, logits = jax.lax.scan(step, (zeros, zeros), inputs)
        return logits
