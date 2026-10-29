"""Deterministic character-level addition; no training or I/O on import."""

from dataclasses import dataclass
import random

import numpy as np
from jaxtyping import Bool, Int

from config import parse_config, thinking_steps


# Retain historical token IDs and checkpoint shapes; only addition is supported.
VOCAB = "_0123456789+-*=$~:"
PAD_ID = VOCAB.index("_")
EOS_ID = VOCAB.index("$")
DEFAULT_THINKING_STEPS = (0, 1, 2, 4, 8, 16)
DATA_DEFAULTS = {
    "max_operand": 19,
    "dataset_size": 120,
    "completion_length": 32,
    "thinking_steps": DEFAULT_THINKING_STEPS,
    "seed": 0,
}


def encode(text: str) -> Int[np.ndarray, "tokens"]:
    return np.asarray([VOCAB.index(char) for char in text], dtype=np.int32)


def decode(tokens: Int[np.ndarray, "tokens"]) -> str:
    return "".join(VOCAB[int(token)] for token in tokens).split("$", 1)[0].replace("_", "")


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    thinking_steps: int = 0

    @property
    def completion(self) -> str:
        return "~" * self.thinking_steps + ":" + self.answer + "$"


@dataclass(frozen=True)
class ArithmeticData:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    sequence_length: int
    completion_length: int
    thinking_steps: tuple[int, ...]


def generate_data(
    *, max_operand: int = 19, seed: int = 0,
    completion_length: int = 32, dataset_size: int = 120,
    thinking_steps: tuple[int, ...] = DEFAULT_THINKING_STEPS,
) -> ArithmeticData:
    """Balance operand-length groups for addition, then randomly split 80/10/10.

    Operands include zero and max_operand.
    Commuted expressions are distinct prompts and may occur in different splits.
    Sample dataset_size unique equations without enumerating the full space.
    Balance (left digits, right digits) quotas within one example,
    redistributing quota from exhausted groups. Zero has one digit; the longest
    range ends at max_operand. Sampling within each group is uniform.

    Shuffle unique equations with seed, then reserve floor(dataset_size / 10) for
    each held-out split. Expand each equation into thinking_steps variants only
    after splitting; these variants do not count against dataset_size.
    Digit-length balance applies to the combined equations, not individual splits.
    Metrics describe digit-balanced interpolation, not uniform integer pairs.
    completion_length must fit the longest thinking prefix, delimiter, answer,
    and EOS.
    """
    if max_operand < 1:
        raise ValueError("max_operand must be at least 1")
    thinking_steps = tuple(thinking_steps)
    if (
        not thinking_steps
        or any(steps < 0 for steps in thinking_steps)
        or any(left >= right for left, right in zip(thinking_steps, thinking_steps[1:]))
    ):
        raise ValueError("thinking_steps must be nonempty, nonnegative, and strictly increasing")
    population_size = (max_operand + 1) ** 2
    if not 10 <= dataset_size <= population_size:
        raise ValueError(
            f"dataset_size must be between 10 and {population_size} unique equations"
        )
    rng = random.Random(seed)
    operand_ranges = [
        (0 if digits == 1 else 10 ** (digits - 1), min(max_operand, 10 ** digits - 1))
        for digits in range(1, len(str(max_operand)) + 1)
    ]
    groups = [
        (a_min, a_max - a_min + 1, b_min, b_max - b_min + 1)
        for a_min, a_max in operand_ranges
        for b_min, b_max in operand_ranges
    ]
    capacities = [a_count * b_count for _, a_count, _, b_count in groups]
    # Find the common quota after filling groups too small to reach it.
    remaining = dataset_size
    quota = 0
    for index, capacity in enumerate(sorted(capacities)):
        quota = remaining // (len(groups) - index)
        if capacity >= quota:
            break
        remaining -= capacity
    counts = [min(capacity, quota) for capacity in capacities]
    extra = dataset_size - sum(counts)
    eligible = [index for index, capacity in enumerate(capacities) if capacity > quota]
    for index in rng.sample(eligible, extra):
        counts[index] += 1

    examples = []
    for (a_min, a_count, b_min, b_count), count in zip(groups, counts):
        for index in rng.sample(range(a_count * b_count), count):
            a_offset, b_offset = divmod(index, b_count)
            a, b = a_min + a_offset, b_min + b_offset
            examples.append(Example(f"{a}+{b}=", str(a + b)))
    required_completion = thinking_steps[-1] + max(len(e.completion) for e in examples)
    if completion_length < required_completion:
        raise ValueError(
            f"completion_length must be at least {required_completion} to fit thinking, "
            "the answer delimiter, answer, and EOS"
        )
    rng.shuffle(examples)
    shuffled = tuple(examples)
    held_out = dataset_size // 10

    def variants(equations):
        return tuple(
            Example(example.prompt, example.answer, steps)
            for example in equations
            for steps in thinking_steps
        )

    return ArithmeticData(
        train=variants(shuffled[: -2 * held_out]),
        validation=variants(shuffled[-2 * held_out : -held_out]),
        test=variants(shuffled[-held_out:]),
        sequence_length=(
            max(len(e.prompt) for e in examples)
            + completion_length
        ),
        completion_length=completion_length,
        thinking_steps=thinking_steps,
    )


def encode_example(example: Example, sequence_length: int) -> dict[str, np.ndarray]:
    """Encode prompt + ~...:answer$ with right padding.

    mask supervises thinking, delimiter, answer, and EOS; answer_mask includes
    only answer characters and EOS.
    Both exclude prompt and padding; losses normalize by supervised token count.
    """
    text = encode(example.prompt + example.completion)
    tokens = np.full(sequence_length, PAD_ID, dtype=np.int32)
    tokens[: len(text)] = text
    positions = np.arange(sequence_length - 1)
    mask: Bool[np.ndarray, "time"] = (
        (positions >= len(example.prompt) - 1) & (positions < len(text) - 1)
    )
    answer_mask = mask & (positions >= len(example.prompt) + example.thinking_steps)
    return {
        "inputs": tokens[:-1], "targets": tokens[1:],
        "mask": mask, "answer_mask": answer_mask,
    }


if __name__ == "__main__":
    import argparse
    import pydoc

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--data-config", metavar="'KEY=VALUE ...'",
        help="Data settings, e.g. 'max-operand=19 dataset-size=120 thinking-steps=0,1,2,4 seed=0'",
    )
    parser.add_argument(
        "--sort", choices=("dataset", "equation", "operand-size"), default="dataset",
        help="Viewer order within each split: sampled order, left/right, or max/min operand",
    )
    args = parser.parse_args()
    sort_order = args.sort
    try:
        overrides = parse_config(args.data_config, DATA_DEFAULTS, "--data-config")
        options = dict(DATA_DEFAULTS)
        for key, value in overrides.items():
            options[key] = thinking_steps(value) if key == "thinking_steps" else int(value)
        data = generate_data(**options)
    except ValueError as error:
        parser.error(str(error))

    def sort_key(example):
        left, right = map(int, example.prompt[:-1].split("+"))
        equation = (left, right)
        if sort_order == "operand-size":
            return (max(left, right), min(left, right), *equation)
        return equation

    thinking_lengths = ", ".join(map(str, data.thinking_steps))
    lines = [f"Thinking steps (~ tokens): {thinking_lengths}", ""]
    for name in ("train", "validation", "test"):
        examples = getattr(data, name)
        equations = tuple({example.prompt: example for example in examples}.values())
        if sort_order != "dataset":
            equations = sorted(equations, key=sort_key)
        lines.append(f"{name}: {len(equations)} equations, {len(examples)} variants")
        lines.extend(f"  {example.prompt}:{example.answer}$" for example in equations)
        lines.append("")
    pydoc.pager("\n".join(lines))
