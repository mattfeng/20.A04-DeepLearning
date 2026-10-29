"""A stacked, chrono-initialized LSTM exposing causal next-token logits."""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from data import VOCAB


class LSTM(eqx.Module):
    embedding: eqx.nn.Embedding
    input_projections: tuple[eqx.nn.Linear, ...]
    recurrent_projections: tuple[eqx.nn.Linear, ...]
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    chrono_max_timescale: float = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    skip_connections: bool = eqx.field(static=True)

    def __init__(
        self, vocab_size: int, hidden_size: int = 128, *,
        embedding_size: int | None = None, chrono_max_timescale: float = 32.0,
        num_layers: int = 1, skip_connections: bool = False, key: PRNGKeyArray,
    ):
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if (
            chrono_max_timescale is None
            or not math.isfinite(chrono_max_timescale)
            or chrono_max_timescale < 2
        ):
            raise ValueError("chrono_max_timescale must be finite and at least 2")
        embedding_key, input_key, recurrent_key, output_key = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.chrono_max_timescale = chrono_max_timescale
        self.embedding = eqx.nn.Embedding(vocab_size, self.embedding_size, key=embedding_key)
        input_projections = []
        recurrent_projections = []
        for layer in range(num_layers):
            layer_key = key if layer == 0 else jax.random.fold_in(key, layer + 4)
            if layer:
                input_key, recurrent_key = jax.random.split(layer_key)
            input_projection = eqx.nn.Linear(
                self.embedding_size if layer == 0 else hidden_size,
                4 * hidden_size, key=input_key,
            )
            recurrent_projections.append(eqx.nn.Linear(
                hidden_size, 4 * hidden_size, use_bias=False, key=recurrent_key,
            ))
            # Gate order: input, forget, candidate, output. Chrono initialization
            # sets forget biases to log(U[1, T_max - 1)); all biases remain trainable.
            bias_key = jax.random.fold_in(layer_key, 4)
            chrono_bias = jnp.log(jax.random.uniform(
                bias_key, (hidden_size,), dtype=input_projection.bias.dtype,
                minval=1.0, maxval=chrono_max_timescale - 1.0,
            ))
            bias = input_projection.bias.at[hidden_size:2 * hidden_size].set(chrono_bias)
            input_projections.append(eqx.tree_at(
                lambda projection: projection.bias, input_projection, bias,
            ))
        self.input_projections = tuple(input_projections)
        self.recurrent_projections = tuple(recurrent_projections)
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
                "num_layers": self.num_layers,
                "skip_connections": self.skip_connections,
                "embedding_size": self.embedding_size,
                "chrono_max_timescale": self.chrono_max_timescale,
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
        # Reconstruct only current settings. Historical initialization metadata
        # does not affect the stored weights, which are restored unchanged.
        model_config = {
            name: value for name, value in config["model_config"].items()
            if name in (
                "hidden_size", "embedding_size", "num_layers",
                "skip_connections", "chrono_max_timescale",
            ) and value is not None
        }
        model = eqx.filter_eval_shape(
            cls, len(VOCAB), **model_config, key=jax.random.key(0),
        )
        return eqx.tree_deserialise_leaves(directory / "model.eqx", model), config

    def __call__(self, tokens: Int[Array, "time"]) -> Float[Array, "time vocab"]:
        """Return causal logits, resetting hidden and cell state for each equation.

        Optional skips add each upper layer's input to its output, leaving
        recurrent hidden/cell states unchanged. The first layer has no skip,
        so its embedding dimension may differ from the hidden dimension.
        """
        inputs = jax.vmap(self.embedding)(tokens)

        def step(state, inputs):
            next_state = []
            for layer, ((hidden, cell), input_projection, recurrent_projection) in enumerate(zip(
                state, self.input_projections, self.recurrent_projections,
            )):
                gates = input_projection(inputs) + recurrent_projection(hidden)
                input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4)
                cell = (
                    jax.nn.sigmoid(forget_gate) * cell
                    + jax.nn.sigmoid(input_gate) * jnp.tanh(candidate)
                )
                hidden = jax.nn.sigmoid(output_gate) * jnp.tanh(cell)
                next_state.append((hidden, cell))
                inputs = inputs + hidden if self.skip_connections and layer > 0 else hidden
            return tuple(next_state), self.readout(inputs)

        zeros = jnp.zeros(self.hidden_size, dtype=inputs.dtype)
        _, logits = jax.lax.scan(step, ((zeros, zeros),) * self.num_layers, inputs)
        return logits
