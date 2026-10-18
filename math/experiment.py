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
    DEFAULT_THINKING_STEPS, EOS_ID, OPERATIONS, PAD_ID, VOCAB, Example, decode, encode,
    addition_curriculum_stages, encode_example, generate_data,
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
            op = next(op for op in OPERATIONS if op in example.prompt)
            left, right = example.prompt[:-1].split(op)
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
    Curriculum stages select cumulative subsets of the fixed training split.
    Training accuracy still covers the full training split at every epoch.
    """
    stages = (
        addition_curriculum_stages(data.train)
        if args.curriculum == "addition" else ()
    )
    if stages and args.epochs < len(stages) * args.curriculum_stage_epochs:
        raise ValueError(
            f"Addition curriculum has {len(stages)} stages and requires at least "
            f"{len(stages) * args.curriculum_stage_epochs} epochs "
            f"at {args.curriculum_stage_epochs} epochs per stage"
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
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
        "curriculum_stages": [
            {
                "max_digits": stage.max_digits,
                "max_carry_chain": stage.max_carry_chain,
                "train_examples": len(stage.examples),
                "train_equations": len(_unique_equations(stage.examples)),
            }
            for stage in stages
        ],
    }
    run = trackio.init(project=args.project, name=args.name, config=config)
    try:
        config["name"] = run.name
        baseline = evaluate(model, data.validation, data, args.batch_size)
        trackio.log({"epoch": 0, **{f"validation/{k}": v for k, v in baseline.items()}})
        print(f"Untrained validation: {baseline}", flush=True)
        for epoch in range(1, args.epochs + 1):
            train_examples = data.train
            curriculum_metrics = {}
            description = f"Epoch {epoch}/{args.epochs}"
            if stages:
                stage_index = min(
                    (epoch - 1) // args.curriculum_stage_epochs, len(stages) - 1,
                )
                stage = stages[stage_index]
                train_examples = stage.examples
                stage_config = config["curriculum_stages"][stage_index]
                curriculum_metrics = {
                    "curriculum/stage": stage_index + 1,
                    **{f"curriculum/{key}": value for key, value in stage_config.items()},
                }
                description += f" [stage {stage_index + 1}/{len(stages)}]"
            total_batches = (len(train_examples) + args.batch_size - 1) // args.batch_size
            weighted_loss = 0.0
            token_count = 0
            with tqdm(
                batches(
                    train_examples, data.sequence_length, args.batch_size,
                    shuffle_seed=args.seed + epoch,
                ),
                total=total_batches, desc=description, unit="batch",
            ) as progress:
                for batch in progress:
                    model, state, value = train_step(model, state, batch, optimizer)
                    count = int(np.sum(batch["mask"]))
                    weighted_loss += float(value) * count
                    token_count += count
                    progress.set_postfix(loss=f"{weighted_loss / token_count:.4f}", refresh=False)
            train_accuracy = _exact_accuracy(model, data.train, data, args.batch_size)
            validation = evaluate(model, data.validation, data, args.batch_size)
            train_loss = weighted_loss / token_count
            trackio.log({
                "epoch": epoch,
                "train/loss": train_loss,
                **curriculum_metrics,
                **{f"train/{k}": v for k, v in train_accuracy.items()},
                **{f"validation/{k}": v for k, v in validation.items()},
            })
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"train exact {train_accuracy['exact_accuracy']:.2%} | "
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
    parser.add_argument(
        "--operations", default=None,
        help="Operators to sample (default: + for addition curriculum, otherwise +-*)",
    )
    parser.add_argument(
        "--curriculum", choices=("none", "addition"), default="none",
        help="Cumulative addition stages ordered by operand digits and carry-chain length",
    )
    parser.add_argument(
        "--curriculum-stage-epochs", type=int, default=5,
        help="Epochs per curriculum stage; remaining epochs use the final stage (default: 5)",
    )
    parser.add_argument("--dataset-size", type=int, default=120)
    parser.add_argument("--completion-length", type=int, default=32)
    parser.add_argument(
        "--thinking-steps", type=int, nargs="+", default=DEFAULT_THINKING_STEPS,
        help="Increasing counts of ~ tokens per equation; variants do not count toward dataset-size",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--num-layers", type=int, default=1,
        help="Number of stacked recurrent layers, each with hidden-size units (default: 1)",
    )
    parser.add_argument(
        "--skip-connections", action="store_true",
        help="Add residual connections around recurrent layers above the first",
    )
    if selection.model == "lstm":
        lstm_options = parser.add_argument_group("LSTM options")
        lstm_options.add_argument(
            "--embedding-size", type=int, default=None,
            help="Token embedding dimension (default: hidden-size)",
        )
        bias_options = lstm_options.add_mutually_exclusive_group()
        bias_options.add_argument(
            "--forget-bias", type=float, default=1.0,
            help="Initial fixed forget-gate bias (default: 1); remains trainable",
        )
        bias_options.add_argument(
            "--chrono-max-timescale", type=float, default=None,
            help="Initialize forget biases as log(U(1, T-1)); finite T >= 2",
        )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW decay for all trainable parameters, including biases (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="arithmetic")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--output-dir", default="artifacts/math",
        help="Parent directory for unique run directories containing final weights and config",
    )
    args = parser.parse_args()
    if args.operations is None:
        args.operations = "+" if args.curriculum == "addition" else "+-*"
    if args.curriculum == "addition" and args.operations != "+":
        parser.error("addition curriculum requires --operations +")
    if args.curriculum_stage_epochs < 1:
        parser.error("curriculum-stage-epochs must be positive")
    if min(args.hidden_size, args.batch_size, args.epochs) < 1 or args.learning_rate <= 0:
        parser.error("hidden-size, batch-size, epochs, and learning-rate must be positive")
    if args.num_layers < 1:
        parser.error("num-layers must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if args.model == "lstm":
        if args.embedding_size is None:
            args.embedding_size = args.hidden_size
        if args.embedding_size < 1:
            parser.error("embedding-size must be positive")
        if not math.isfinite(args.forget_bias):
            parser.error("forget-bias must be finite")
        if args.chrono_max_timescale is not None and (
            not math.isfinite(args.chrono_max_timescale) or args.chrono_max_timescale < 2
        ):
            parser.error("chrono-max-timescale must be finite and at least 2")
    return args


def main():
    args = parse_args()
    data = generate_data(
        max_operand=args.max_operand, operations=args.operations, seed=args.seed,
        completion_length=args.completion_length,
        dataset_size=args.dataset_size,
        thinking_steps=args.thinking_steps,
    )
    model_options = {
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "skip_connections": args.skip_connections,
    }
    if args.model == "lstm":
        model_options.update(
            embedding_size=args.embedding_size, forget_bias=args.forget_bias,
            chrono_max_timescale=args.chrono_max_timescale,
        )
    model_class = {"rnn": RNN, "lstm": LSTM}[args.model]
    model = model_class(len(VOCAB), **model_options, key=jax.random.key(args.seed))
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
