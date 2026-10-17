"""A vanilla tanh RNN exposing token-sequence -> next-token logits."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from data import VOCAB


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

    def save_artifact(self, config: dict, output_dir: str | Path) -> Path:
        """Save under config['name'] (the Trackio run name); never overwrite a run."""
        run_name = config["name"]
        if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("Artifact run name must be a single directory name")
        directory = Path(output_dir) / run_name
        directory.mkdir(parents=True)
        config = {
            **config,
            "architecture": type(self).__name__,
            "model_config": {"hidden_size": self.hidden_size},
            "vocab": VOCAB,
        }
        eqx.tree_serialise_leaves(directory / "model.eqx", self)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return directory

    @classmethod
    def load_artifact(cls, directory: str | Path) -> tuple["RNN", dict]:
        """Return (model, config) from saved RNN inference weights and metadata."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config["vocab"] != VOCAB:
            raise ValueError("Artifact vocabulary does not match the current vocabulary")
        if config["architecture"] != cls.__name__:
            raise ValueError("Artifact architecture does not match this model")
        model = eqx.filter_eval_shape(
            cls, len(VOCAB), config["model_config"]["hidden_size"],
            key=jax.random.key(0),
        )
        return eqx.tree_deserialise_leaves(directory / "model.eqx", model), config

    def __call__(self, tokens: Int[Array, "time"]) -> Float[Array, "time vocab"]:
        """Return causal logits, resetting hidden state for each equation."""
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
