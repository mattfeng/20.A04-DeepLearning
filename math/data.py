"""Deterministic character-level arithmetic; no training or I/O on import."""

from dataclasses import dataclass
import operator
import random

import numpy as np
from jaxtyping import Bool, Int


VOCAB = "_0123456789+-*=$~:"
PAD_ID = VOCAB.index("_")
EOS_ID = VOCAB.index("$")
OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def encode(text: str) -> Int[np.ndarray, "tokens"]:
    return np.asarray([VOCAB.index(char) for char in text], dtype=np.int32)


def decode(tokens: Int[np.ndarray, "tokens"]) -> str:
    return "".join(VOCAB[int(token)] for token in tokens).split("$", 1)[0].replace("_", "")


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str

    @property
    def completion(self) -> str:
        return ":" + self.answer + "$"


@dataclass(frozen=True)
class ArithmeticData:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    sequence_length: int
    completion_length: int


def generate_data(
    *, max_operand: int = 19, operations: str = "+-*", seed: int = 0,
    completion_length: int = 32, dataset_size: int = 120,
) -> ArithmeticData:
    """Balance operator/operand-length groups, then randomly split 80/10/10.

    Operands include zero and max_operand. Subtraction may be negative.
    Commuted expressions are distinct prompts and may occur in different splits.
    Sample dataset_size unique equations, balancing both operands' digit lengths.
    Exhausted groups donate their quota to the remaining groups.
    """
    if max_operand < 1:
        raise ValueError("max_operand must be at least 1")
    if not operations or len(set(operations)) != len(operations):
        raise ValueError("operations must be nonempty and contain no duplicates")
    if any(op not in OPERATIONS for op in operations):
        raise ValueError("supported operations are +, -, and *")
    operand_count = max_operand + 1
    pairs_per_operation = operand_count ** 2
    population_size = len(operations) * pairs_per_operation
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
        (op, a_min, a_max - a_min + 1, b_min, b_max - b_min + 1)
        for op in operations
        for a_min, a_max in operand_ranges
        for b_min, b_max in operand_ranges
    ]
    capacities = [a_count * b_count for _, _, a_count, _, b_count in groups]
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
    for (op, a_min, a_count, b_min, b_count), count in zip(groups, counts):
        for index in rng.sample(range(a_count * b_count), count):
            a_offset, b_offset = divmod(index, b_count)
            a, b = a_min + a_offset, b_min + b_offset
            examples.append(Example(f"{a}{op}{b}=", str(OPERATIONS[op](a, b))))
    if completion_length < max(len(e.completion) for e in examples):
        raise ValueError("completion_length must fit the answer delimiter, answer, and EOS")
    rng.shuffle(examples)
    shuffled = tuple(examples)
    held_out = dataset_size // 10
    return ArithmeticData(
        train=shuffled[: -2 * held_out],
        validation=shuffled[-2 * held_out : -held_out],
        test=shuffled[-held_out:],
        sequence_length=(
            max(len(e.prompt) for e in examples)
            + completion_length
        ),
        completion_length=completion_length,
    )


def encode_example(example: Example, sequence_length: int) -> dict[str, np.ndarray]:
    """Supervise the completion; expose a second mask for answer-only metrics."""
    text = encode(example.prompt + example.completion)
    tokens = np.full(sequence_length, PAD_ID, dtype=np.int32)
    tokens[: len(text)] = text
    positions = np.arange(sequence_length - 1)
    mask: Bool[np.ndarray, "time"] = (
        (positions >= len(example.prompt) - 1) & (positions < len(text) - 1)
    )
    answer_mask = mask & (positions >= len(example.prompt))
    return {
        "inputs": tokens[:-1], "targets": tokens[1:],
        "mask": mask, "answer_mask": answer_mask,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-operand", type=int, default=19)
    parser.add_argument("--operations", default="+-*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=120)
    parser.add_argument("--completion-length", type=int, default=32)
    args = parser.parse_args()
    data = generate_data(**vars(args))
    for name in ("train", "validation", "test"):
        examples = getattr(data, name)
        print(f"{name}: {len(examples)} equations")
        for example in examples[:5]:
            print(f"  {example.prompt}{example.completion}")
