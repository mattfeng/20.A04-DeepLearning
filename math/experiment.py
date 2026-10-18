"""Train and compare sequence models on a fixed arithmetic completion task."""

import argparse
import math

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

from data import (
    DEFAULT_THINKING_STEPS, EOS_ID, PAD_ID, VOCAB, Example, decode, encode,
    encode_example, generate_data,
)
from lstm_model import LSTM
from rnn_model import RNN


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


@eqx.filter_jit
def train_step(model, state, batch, optimizer):
    value, gradients = eqx.filter_value_and_grad(loss)(
        model, batch["inputs"], batch["targets"], batch["mask"]
    )
    updates, state = optimizer.update(
        gradients, state, eqx.filter(model, eqx.is_inexact_array)
    )
    return eqx.apply_updates(model, updates), state, value


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
    correct = 0
    equations = _unique_equations(examples)
    for start in range(0, len(equations), batch_size):
        subset = equations[start : start + batch_size]
        prompts, lengths = completion_batch(subset, data.sequence_length)
        predictions = np.asarray(complete(model, prompts, lengths, data.completion_length))
        for example, prediction in zip(subset, predictions):
            completion = "".join(VOCAB[int(token)] for token in prediction[len(example.prompt):])
            correct += extract_answer(completion) == example.answer
    return {"loss": weighted_loss / token_count, "exact_accuracy": correct / len(equations)}


def run_experiment(model: eqx.Module, data, args):
    """Train a deterministic causal Equinox model; keep data and splits shared.

    The shared-vocabulary contract is integer tokens (time,) -> raw logits
    (time, vocab), plus model.save_artifact(config, output_dir). Models own their
    serialization and reconstruction metadata. For architecture comparisons, fix
    seed, dataset size, operand range, operations, splits, thinking budgets, limit,
    schedule, and metrics.

    Log untrained validation at epoch 0, then training/validation each epoch.
    Training shuffles with seed + epoch; evaluation order stays deterministic.
    Trackio logs locally; use distinct run names. Save final inference artifacts
    before test evaluation; optimizer state is not saved.
    """
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
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
        print(f"Untrained validation: {baseline}", flush=True)
        total_batches = (len(data.train) + args.batch_size - 1) // args.batch_size
        for epoch in range(1, args.epochs + 1):
            weighted_loss = 0.0
            token_count = 0
            with tqdm(
                batches(
                    data.train, data.sequence_length, args.batch_size,
                    shuffle_seed=args.seed + epoch,
                ),
                total=total_batches, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
            ) as progress:
                for batch in progress:
                    model, state, value = train_step(model, state, batch, optimizer)
                    count = int(np.sum(batch["mask"]))
                    weighted_loss += float(value) * count
                    token_count += count
                    progress.set_postfix(loss=f"{weighted_loss / token_count:.4f}", refresh=False)
            validation = evaluate(model, data.validation, data, args.batch_size)
            train_loss = weighted_loss / token_count
            trackio.log({
                "epoch": epoch,
                "train/loss": train_loss,
                **{f"validation/{k}": v for k, v in validation.items()},
            })
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
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


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model", choices=("rnn", "lstm"), default="rnn")
    selection, _ = selector.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[selector])
    parser.add_argument("--max-operand", type=int, default=19)
    parser.add_argument("--operations", default="+-*")
    parser.add_argument("--dataset-size", type=int, default=120)
    parser.add_argument("--completion-length", type=int, default=32)
    parser.add_argument(
        "--thinking-steps", type=int, nargs="+", default=DEFAULT_THINKING_STEPS,
        help="Increasing counts of ~ tokens per equation; variants do not count toward dataset-size",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    if selection.model == "lstm":
        lstm_options = parser.add_argument_group("LSTM options")
        lstm_options.add_argument(
            "--embedding-size", type=int, default=None,
            help="Token embedding dimension (default: hidden-size)",
        )
        lstm_options.add_argument(
            "--forget-bias", type=float, default=1.0,
            help="Initial forget-gate bias; trainable after initialization (default: 1)",
        )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="arithmetic")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--output-dir", default="artifacts/math",
        help="Parent directory for unique run directories containing final weights and config",
    )
    args = parser.parse_args()
    if min(args.hidden_size, args.batch_size, args.epochs) < 1 or args.learning_rate <= 0:
        parser.error("hidden-size, batch-size, epochs, and learning-rate must be positive")
    if args.model == "lstm":
        if args.embedding_size is None:
            args.embedding_size = args.hidden_size
        if args.embedding_size < 1:
            parser.error("embedding-size must be positive")
        if not math.isfinite(args.forget_bias):
            parser.error("forget-bias must be finite")
    return args


def main():
    args = parse_args()
    data = generate_data(
        max_operand=args.max_operand, operations=args.operations, seed=args.seed,
        completion_length=args.completion_length,
        dataset_size=args.dataset_size,
        thinking_steps=args.thinking_steps,
    )
    model_options = {"hidden_size": args.hidden_size}
    if args.model == "lstm":
        model_options.update(embedding_size=args.embedding_size, forget_bias=args.forget_bias)
    model_class = {"rnn": RNN, "lstm": LSTM}[args.model]
    model = model_class(len(VOCAB), **model_options, key=jax.random.key(args.seed))
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
