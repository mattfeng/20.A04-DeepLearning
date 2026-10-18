"""A stacked vanilla tanh RNN exposing token-sequence -> next-token logits."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from data import VOCAB


class RNN(eqx.Module):
    embedding: eqx.nn.Embedding
    input_projections: tuple[eqx.nn.Linear, ...]
    recurrent_projections: tuple[eqx.nn.Linear, ...]
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    skip_connections: bool = eqx.field(static=True)

    def __init__(
        self, vocab_size: int, hidden_size: int = 128, *,
        num_layers: int = 1, skip_connections: bool = False, key: PRNGKeyArray,
    ):
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        embedding_key, input_key, recurrent_key, output_key = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=embedding_key)
        input_projections = []
        recurrent_projections = []
        for layer in range(num_layers):
            if layer:
                input_key, recurrent_key = jax.random.split(jax.random.fold_in(key, layer + 4))
            input_projections.append(eqx.nn.Linear(hidden_size, hidden_size, key=input_key))
            recurrent_projections.append(eqx.nn.Linear(
                hidden_size, hidden_size, use_bias=False, key=recurrent_key,
            ))
        self.input_projections = tuple(input_projections)
        self.recurrent_projections = tuple(recurrent_projections)
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
            "model_config": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "skip_connections": self.skip_connections,
            },
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
            cls, len(VOCAB), **config["model_config"], key=jax.random.key(0),
        )
        return eqx.tree_deserialise_leaves(directory / "model.eqx", model), config

    def __call__(self, tokens: Int[Array, "time"]) -> Float[Array, "time vocab"]:
        """Return causal logits, resetting hidden state for each equation.

        Optional skips add each upper layer's input to its output, never its
        recurrent hidden state. The first layer has no skip connection.
        """
        inputs = jax.vmap(self.embedding)(tokens)

        def step(state, inputs):
            next_state = []
            for layer, (hidden, input_projection, recurrent_projection) in enumerate(zip(
                state, self.input_projections, self.recurrent_projections,
            )):
                hidden = jnp.tanh(input_projection(inputs) + recurrent_projection(hidden))
                next_state.append(hidden)
                inputs = inputs + hidden if self.skip_connections and layer > 0 else hidden
            return tuple(next_state), self.readout(inputs)

        zeros = jnp.zeros(self.hidden_size, dtype=inputs.dtype)
        _, logits = jax.lax.scan(step, (zeros,) * self.num_layers, inputs)
        return logits
