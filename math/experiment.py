"""Train and compare sequence models on a fixed addition completion task."""

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
from typing import Any

import einops
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import trackio
from jaxtyping import Array, Bool, Float, Int
from tqdm import tqdm

from config import parse_config, thinking_steps
from data import (
    DATA_DEFAULTS, EOS_ID, PAD_ID, VOCAB, Example, decode, encode,
    encode_example, generate_data,
)
from lstm_model import LSTM
from rnn_model import RNN
from transformer_model import Transformer


def batches(examples, sequence_length, batch_size, *, shuffle_seed=None):
    source = [encode_example(example, sequence_length) for example in examples]
    dataset = grain.MapDataset.source(source)
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    return dataset.batch(batch_size, drop_remainder=False).to_iter_dataset()


@eqx.filter_jit
def loss(
    model: eqx.Module,
    inputs: Int[Array, "batch time"],
    targets: Int[Array, "batch time"],
    mask: Bool[Array, "batch time"],
) -> Float[Array, ""]:
    logits = jax.vmap(model)(inputs)
    # Flatten batch/time identically for logits, labels, and the selected mask.
    logits = einops.rearrange(logits, "batch time vocab -> (batch time) vocab")
    targets = einops.rearrange(targets, "batch time -> (batch time)")
    weights = einops.rearrange(mask, "batch time -> (batch time)")
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(losses * weights) / jnp.sum(weights)


def _regularization_loss(model, weight_decay, weight_decay_type):
    """Summed parameter penalty, including biases; L1 uses zero subgradient at zero."""
    if weight_decay == 0:
        return jnp.array(0.0)
    parameters = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    if weight_decay_type == "l1":
        return weight_decay * sum(
            jnp.sum(jnp.where(parameter == 0, 0, jnp.abs(parameter)))
            for parameter in parameters
        )
    if weight_decay_type == "l2":
        return 0.5 * weight_decay * sum(jnp.sum(parameter**2) for parameter in parameters)
    raise ValueError("weight_decay_type must be l1 or l2")


@eqx.filter_jit
def train_step(model, state, batch, optimizer, weight_decay, weight_decay_type):
    def objective(model):
        value = loss(model, batch["inputs"], batch["targets"], batch["mask"])
        penalty = _regularization_loss(model, weight_decay, weight_decay_type)
        return value + penalty, (value, penalty)

    (_, (value, penalty)), gradients = eqx.filter_value_and_grad(objective, has_aux=True)(model)
    updates, state = optimizer.update(
        gradients, state, eqx.filter(model, eqx.is_inexact_array)
    )
    return eqx.apply_updates(model, updates), state, value, penalty


@eqx.filter_jit
def complete(model, prompts, prompt_lengths, completion_length):
    """Greedily decode to EOS or completion_length, including intermediate text.

    Recompute sequence logits per token; no architecture-specific cache contract.
    """
    rows = jnp.arange(prompts.shape[0])

    def step(index, carry):
        tokens, finished = carry
        positions = prompt_lengths - 1 + index
        logits = jax.vmap(model)(tokens)
        next_tokens = jnp.argmax(logits[rows, positions], axis=-1)
        next_tokens = jnp.where(finished, PAD_ID, next_tokens)
        tokens = tokens.at[rows, positions + 1].set(next_tokens, mode="drop")
        return tokens, finished | (next_tokens == EOS_ID)

    tokens, _ = jax.lax.fori_loop(
        0, completion_length, step,
        (prompts, jnp.zeros(prompts.shape[0], dtype=jnp.bool_)),
    )
    return tokens


def completion_batch(examples: tuple[Example, ...], sequence_length: int):
    prompts = np.full((len(examples), sequence_length), PAD_ID, dtype=np.int32)
    lengths = np.asarray([len(e.prompt) for e in examples], dtype=np.int32)
    for row, example in enumerate(examples):
        prompts[row, : lengths[row]] = encode(example.prompt)
    return prompts, lengths


def extract_answer(completion: str) -> str | None:
    """Score the last :answer before the first EOS; require both delimiters.

    Ignore intermediate content: 1+1=0:9:2$ scores as 2. Extra answer characters
    remain part of the answer and fail exact matching.
    """
    body, eos, _ = completion.partition("$")
    _, separator, answer = body.rpartition(":")
    return answer if separator and eos else None


def _unique_equations(examples):
    """Collapse thinking variants while preserving the original equation order."""
    return tuple({example.prompt: example for example in examples}.values())


def _exact_accuracy(model, examples, data, batch_size):
    """Overall and operand-digit-pair accuracy, counting each unique equation once."""
    correct = 0
    group_counts = {}
    equations = _unique_equations(examples)
    for start in range(0, len(equations), batch_size):
        subset = equations[start : start + batch_size]
        prompts, lengths = completion_batch(subset, data.sequence_length)
        predictions = np.asarray(complete(model, prompts, lengths, data.completion_length))
        for example, prediction in zip(subset, predictions):
            completion = "".join(VOCAB[int(token)] for token in prediction[len(example.prompt):])
            is_correct = extract_answer(completion) == example.answer
            correct += is_correct
            left, right = example.prompt[:-1].split("+")
            group = (len(left), len(right))
            group_correct, group_total = group_counts.get(group, (0, 0))
            group_counts[group] = (group_correct + is_correct, group_total + 1)
    return {
        "exact_accuracy": correct / len(equations),
        **{
            f"exact_accuracy/operand_lengths/{left}x{right}": hits / total
            for (left, right), (hits, total) in sorted(group_counts.items())
        },
    }


def evaluate(model, examples, data, batch_size):
    """Answer/EOS loss across variants; greedy exact accuracy per unique equation."""
    weighted_loss = 0.0
    token_count = 0
    for batch in batches(examples, data.sequence_length, batch_size):
        count = int(np.sum(batch["answer_mask"]))
        weighted_loss += float(
            loss(model, batch["inputs"], batch["targets"], batch["answer_mask"])
        ) * count
        token_count += count
    return {
        "loss": weighted_loss / token_count,
        **_exact_accuracy(model, examples, data, batch_size),
    }


def train_epoch(model, state, optimizer, data, args, epoch):
    """Train on the full split, averaging losses by supervised token count."""
    weighted_loss = 0.0
    weighted_penalty = 0.0
    token_count = 0
    total_batches = (len(data.train) + args.batch_size - 1) // args.batch_size
    with tqdm(
        batches(
            data.train, data.sequence_length, args.batch_size,
            shuffle_seed=args.seed + epoch,
        ),
        total=total_batches, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
    ) as progress:
        for batch in progress:
            model, state, value, penalty = train_step(
                model, state, batch, optimizer,
                args.weight_decay, args.weight_decay_type,
            )
            count = int(np.sum(batch["mask"]))
            weighted_loss += float(value) * count
            weighted_penalty += float(penalty) * count
            token_count += count
            progress.set_postfix(
                loss=f"{weighted_loss / token_count:.4f}",
                total=f"{(weighted_loss + weighted_penalty) / token_count:.4f}",
                refresh=False,
            )
    train_loss = weighted_loss / token_count
    train_penalty = weighted_penalty / token_count
    return model, state, {
        "loss": train_loss,
        "regularization_loss": train_penalty,
        "total_loss": train_loss + train_penalty,
    }


def run_experiment(model: eqx.Module, data, args):
    """Log baseline, train/evaluate each epoch, save weights, then evaluate test.

    Training shuffles with seed + epoch; evaluation order stays deterministic.
    Artifacts contain final inference weights and config, not optimizer state.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.learning_rate),
    )
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    config = {
        **vars(args),
        "architecture": type(model).__name__,
        "thinking_steps": data.thinking_steps,
        "train_equations": len(data.train) // len(data.thinking_steps),
        "validation_equations": len(data.validation) // len(data.thinking_steps),
        "test_equations": len(data.test) // len(data.thinking_steps),
        "train_examples": len(data.train),
        "validation_examples": len(data.validation),
        "test_examples": len(data.test),
        "sequence_length": data.sequence_length,
        "completion_length": data.completion_length,
        "vocab": VOCAB,
        "gradient_clip_norm": 1.0,
        "backend": jax.default_backend(),
    }
    run = trackio.init(project=args.project, name=args.name, config=config)
    try:
        config["name"] = run.name
        baseline = evaluate(model, data.validation, data, args.batch_size)
        trackio.log({"epoch": 0, **{f"validation/{k}": v for k, v in baseline.items()}})
        print(f"Initial validation: {baseline}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model, state, train = train_epoch(model, state, optimizer, data, args, epoch)
            train.update(_exact_accuracy(model, data.train, data, args.batch_size))
            validation = evaluate(model, data.validation, data, args.batch_size)
            trackio.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train.items()},
                **{f"validation/{k}": v for k, v in validation.items()},
            })
            print(
                f"Epoch {epoch:03d} | train loss {train['loss']:.4f} | "
                f"train total {train['total_loss']:.4f} | "
                f"train exact {train['exact_accuracy']:.2%} | "
                f"validation loss {validation['loss']:.4f} | "
                f"validation exact {validation['exact_accuracy']:.2%}",
                flush=True,
            )
        directory = model.save_artifact(config, args.output_dir)
        print(f"Saved model artifact: {directory}", flush=True)
        # Test labels are evaluated only after the fixed training schedule.
        test = evaluate(model, data.test, data, args.batch_size)
        trackio.log({f"test/{k}": v for k, v in test.items()})
        print(f"Test: {test}", flush=True)
        examples = _unique_equations(data.validation)[:5]
        prompts, lengths = completion_batch(examples, data.sequence_length)
        predictions = np.asarray(complete(model, prompts, lengths, data.completion_length))
        for example, prediction in zip(examples, predictions):
            print(f"{example.prompt} -> {decode(prediction)[len(example.prompt):]} "
                  f"(expected {example.answer})")
    finally:
        trackio.finish()
    return model


def _prompt(
    option, default, convert: Callable[[str], Any] = str, *,
    optional=False, minimum=None, positive=False, choices=(),
):
    label = option.replace("_", " ").capitalize()
    if choices:
        label += " (" + "/".join(choices) + ")"
    if convert is bool:
        label += " (yes/no)"
    if default is None:
        shown = "none"
    elif isinstance(default, bool):
        shown = "yes" if default else "no"
    elif isinstance(default, (tuple, list)):
        shown = ",".join(map(str, default))
    else:
        shown = str(default)
    while True:
        text = input(f"{label} [{shown}]: ").strip()
        if not text:
            text = shown
        if optional and text.lower() == "none":
            return None
        try:
            if convert is bool:
                value = {"yes": True, "true": True, "no": False, "false": False}[text.lower()]
            else:
                value = convert(text)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("must be finite")
            if positive and value <= 0:
                raise ValueError("must be positive")
            if minimum is not None and value < minimum:
                raise ValueError(f"must be at least {minimum}")
            if choices and value not in choices:
                raise ValueError("choose " + ", ".join(choices))
            if convert is str and not value:
                raise ValueError("must not be empty")
        except (ValueError, KeyError) as error:
            print(f"Invalid {option}: {error}. Please try again.")
        else:
            return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--model-config", metavar="'KEY=VALUE ...'",
        help="Model prompt defaults, e.g. 'model=lstm hidden-size=128 num-layers=3'",
    )
    parser.add_argument(
        "--data-config", metavar="'KEY=VALUE ...'",
        help="Data prompt defaults, e.g. 'max-operand=19 dataset-size=120 seed=0'",
    )
    parser.add_argument(
        "--experiment-config", metavar="'KEY=VALUE ...'",
        help="Training/run prompt defaults, e.g. 'epochs=50 seed=0'",
    )
    cli = parser.parse_args()
    model_defaults = {
        "init_from": None,
        "model": "rnn",
        "hidden_size": 128,
        "num_layers": 1,
        "skip_connections": False,
        "embedding_size": None,
        "chrono_max_timescale": 32.0,
        "num_heads": 4,
        "feedforward_size": None,
    }
    experiment_defaults = {
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "weight_decay_type": "l2",
        "seed": 0,
        "project": "arithmetic",
        "name": None,
        "output_dir": "artifacts/math",
    }

    try:
        model_config = parse_config(cli.model_config, model_defaults, "--model-config")
        data_config = parse_config(cli.data_config, DATA_DEFAULTS, "--data-config")
        experiment_config = parse_config(
            cli.experiment_config, experiment_defaults, "--experiment-config",
        )
    except ValueError as error:
        parser.error(str(error))
    data_defaults = {**DATA_DEFAULTS, **data_config}
    data_defaults["data_seed"] = data_defaults.pop("seed")
    defaults = {
        **model_defaults, **model_config,
        **data_defaults,
        **experiment_defaults, **experiment_config,
    }
    args = argparse.Namespace()

    def ask(option, convert: Callable[[str], Any] = str, **kwargs):
        value = _prompt(option, defaults[option], convert, **kwargs)
        setattr(args, option, value)
        return value

    print("Customize the experiment. Enter keeps [default]; none clears optional values.")
    try:
        print("\nModel configuration")
        ask("init_from", optional=True)
        if args.init_from is not None:
            architecture_options = {
                "model", "hidden_size", "num_layers", "skip_connections",
                "embedding_size", "chrono_max_timescale",
                "num_heads", "feedforward_size",
            }
            overrides = architecture_options & model_config.keys()
            if overrides:
                parser.error(
                    "init_from uses the saved architecture; omit "
                    + ", ".join(sorted(overrides))
                )
            print("Using the saved architecture; only data and training settings are editable.")
        else:
            ask("model", choices=("rnn", "lstm", "transformer"))
            ask("hidden_size", int, minimum=1)
            ask("num_layers", int, minimum=1)
            if args.model == "transformer":
                while True:
                    ask("num_heads", int, minimum=1)
                    if args.hidden_size % args.num_heads == 0:
                        break
                    print("Invalid num_heads: must divide hidden_size. Please try again.")
                if defaults["feedforward_size"] is None:
                    defaults["feedforward_size"] = 4 * args.hidden_size
                ask("feedforward_size", int, minimum=1)
            else:
                ask("skip_connections", bool)
            if args.model == "lstm":
                if defaults["embedding_size"] is None:
                    defaults["embedding_size"] = args.hidden_size
                ask("embedding_size", int, minimum=1)
                ask("chrono_max_timescale", float, minimum=2)
        print("\nData configuration")
        ask("max_operand", int, minimum=1)
        ask("dataset_size", int, minimum=10)
        ask("completion_length", int, minimum=1)
        ask("thinking_steps", thinking_steps)
        ask("data_seed", int)
        print("\nTraining and run configuration")
        ask("batch_size", int, minimum=1)
        ask("epochs", int, minimum=1)
        ask("learning_rate", float, positive=True)
        ask("weight_decay", float, minimum=0)
        ask("weight_decay_type", choices=("l1", "l2"))
        ask("seed", int)
        ask("project")
        ask("name", optional=True)
        ask("output_dir")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nConfiguration cancelled; no training started.") from None
    return args


def create_model(args):
    """Load saved architecture and weights, or initialize the selected model."""
    if args.init_from is not None:
        args.init_from = str(Path(args.init_from).expanduser().resolve())
        try:
            saved_config = json.loads((Path(args.init_from) / "config.json").read_text())
            model_class = {
                "RNN": RNN, "LSTM": LSTM, "Transformer": Transformer,
            }.get(saved_config["architecture"])
            if model_class is None:
                raise ValueError(f"Unsupported artifact architecture: {saved_config['architecture']}")
            model, _ = model_class.load_artifact(args.init_from)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"Could not load initial weights: {error}") from error
        args.model = type(model).__name__.lower()
        for option in ("hidden_size", "num_layers"):
            setattr(args, option, getattr(model, option))
        if args.model == "transformer":
            for option in ("num_heads", "feedforward_size"):
                setattr(args, option, getattr(model, option))
        else:
            args.skip_connections = model.skip_connections
        if args.model == "lstm":
            for option in ("embedding_size", "chrono_max_timescale"):
                setattr(args, option, getattr(model, option))
        print(f"Loaded initial weights from {args.init_from}", flush=True)
    else:
        model_options = {
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        }
        if args.model == "transformer":
            model_options.update(
                num_heads=args.num_heads, feedforward_size=args.feedforward_size,
            )
        else:
            model_options["skip_connections"] = args.skip_connections
        if args.model == "lstm":
            model_options.update(
                embedding_size=args.embedding_size,
                chrono_max_timescale=args.chrono_max_timescale,
            )
        model_class = {"rnn": RNN, "lstm": LSTM, "transformer": Transformer}[args.model]
        model = model_class(len(VOCAB), **model_options, key=jax.random.key(args.seed))
    return model


def main():
    args = parse_args()
    model = create_model(args)
    data = generate_data(
        max_operand=args.max_operand, seed=args.data_seed,
        completion_length=args.completion_length,
        dataset_size=args.dataset_size,
        thinking_steps=args.thinking_steps,
    )
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
