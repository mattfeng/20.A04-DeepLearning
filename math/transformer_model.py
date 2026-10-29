"""A decoder-only Transformer exposing token-sequence -> causal next-token logits."""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from data import VOCAB


class TransformerBlock(eqx.Module):
    attention_norm: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    feedforward_norm: eqx.nn.LayerNorm
    feedforward_in: eqx.nn.Linear
    feedforward_out: eqx.nn.Linear

    def __init__(
        self, hidden_size: int, num_heads: int, feedforward_size: int, *, key: PRNGKeyArray,
    ):
        attention_key, input_key, output_key = jax.random.split(key, 3)
        self.attention_norm = eqx.nn.LayerNorm(hidden_size)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads, hidden_size, dropout_p=0.0, key=attention_key,
        )
        self.feedforward_norm = eqx.nn.LayerNorm(hidden_size)
        self.feedforward_in = eqx.nn.Linear(hidden_size, feedforward_size, key=input_key)
        self.feedforward_out = eqx.nn.Linear(feedforward_size, hidden_size, key=output_key)

    def __call__(
        self, inputs: Float[Array, "time hidden"], mask: Bool[Array, "time time"],
    ) -> Float[Array, "time hidden"]:
        normalized = jax.vmap(self.attention_norm)(inputs)
        inputs = inputs + self.attention(normalized, normalized, normalized, mask=mask)
        normalized = jax.vmap(self.feedforward_norm)(inputs)
        hidden = jax.nn.gelu(jax.vmap(self.feedforward_in)(normalized))
        return inputs + jax.vmap(self.feedforward_out)(hidden)


class Transformer(eqx.Module):
    embedding: eqx.nn.Embedding
    blocks: tuple[TransformerBlock, ...]
    final_norm: eqx.nn.LayerNorm
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    feedforward_size: int = eqx.field(static=True)

    def __init__(
        self, vocab_size: int, hidden_size: int = 128, *, num_layers: int = 1,
        num_heads: int = 4, feedforward_size: int | None = None, key: PRNGKeyArray,
    ):
        if feedforward_size is None:
            feedforward_size = 4 * hidden_size
        for name, value in (
            ("hidden_size", hidden_size), ("num_layers", num_layers),
            ("num_heads", num_heads), ("feedforward_size", feedforward_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size
        keys = jax.random.split(key, num_layers + 2)
        self.embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=keys[0])
        self.blocks = tuple(
            TransformerBlock(hidden_size, num_heads, feedforward_size, key=layer_key)
            for layer_key in keys[1:-1]
        )
        self.final_norm = eqx.nn.LayerNorm(hidden_size)
        self.readout = eqx.nn.Linear(hidden_size, vocab_size, key=keys[-1])

    def save_artifact(self, config: dict, output_dir: str | Path) -> Path:
        """Save inference weights and reconstruction settings under the Trackio run name."""
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
                "num_heads": self.num_heads,
                "feedforward_size": self.feedforward_size,
            },
            "vocab": VOCAB,
        }
        eqx.tree_serialise_leaves(directory / "model.eqx", self)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return directory

    @classmethod
    def load_artifact(cls, directory: str | Path) -> tuple["Transformer", dict]:
        """Return (model, config) from saved Transformer inference weights and metadata."""
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
        """Use sinusoidal positions and causal attention, with no dropout or length cap.

        Data are right-padded: padding cannot affect logits at earlier real tokens.
        Positions are computed for the input shape, including longer REPL prompts.
        """
        inputs = jax.vmap(self.embedding)(tokens) * math.sqrt(self.hidden_size)
        positions = jnp.arange(tokens.shape[0], dtype=inputs.dtype)[:, None]
        dimensions = jnp.arange(self.hidden_size)
        frequencies = jnp.exp(-math.log(10000.0) * (2 * (dimensions // 2)) / self.hidden_size)
        angles = positions * frequencies[None, :]
        inputs = inputs + jnp.where(dimensions % 2 == 0, jnp.sin(angles), jnp.cos(angles))
        mask = jnp.tril(jnp.ones((tokens.shape[0], tokens.shape[0]), dtype=jnp.bool_))
        for block in self.blocks:
            inputs = block(inputs, mask)
        return jax.vmap(self.readout)(jax.vmap(self.final_norm)(inputs))
