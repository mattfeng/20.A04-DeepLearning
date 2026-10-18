"""Deterministic character-level arithmetic; no training or I/O on import."""

from dataclasses import dataclass
import operator
import random
import re

import numpy as np
from jaxtyping import Bool, Int


VOCAB = "_0123456789+-*=$~:"
PAD_ID = VOCAB.index("_")
EOS_ID = VOCAB.index("$")
OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
DEFAULT_THINKING_STEPS = (0, 1, 2, 4, 8, 16)


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
class AdditionStage:
    max_digits: int
    max_carry_chain: int
    examples: tuple[Example, ...]


def addition_curriculum_stages(
    examples: tuple[Example, ...],
) -> tuple[AdditionStage, ...]:
    """Build cumulative addition stages from the supplied training examples.

    Difficulty is (maximum operand digit count, longest consecutive outgoing
    carry chain), ordered lexicographically. Stage thresholds therefore retain
    all shorter operand lengths, even when their carry chains are longer.
    Only observed difficulty keys produce stages. Zero has one digit, and a
    carry out of the most significant operand column counts toward the chain.

    Preserve input order and every thinking variant within each stage, without
    duplicating samples. Only prompts are inspected; answers are not used.
    Raise ValueError for empty input or prompts other than nonnegative integer
    addition in the form "a+b=".
    """
    examples = tuple(examples)
    if not examples:
        raise ValueError("addition curriculum requires nonempty examples")

    prompt_keys: dict[str, tuple[int, int]] = {}
    keys = []
    for example in examples:
        if example.prompt not in prompt_keys:
            match = re.fullmatch(r"([0-9]+)\+([0-9]+)=", example.prompt)
            if match is None:
                raise ValueError(
                    f"addition curriculum requires addition prompts: {example.prompt!r}"
                )
            left, right = (operand.lstrip("0") or "0" for operand in match.groups())
            digits = max(len(left), len(right))
            carry = run = longest = 0
            for a, b in zip(reversed(left.zfill(digits)), reversed(right.zfill(digits))):
                carry = (int(a) + int(b) + carry) // 10
                run = run + 1 if carry else 0
                longest = max(longest, run)
            prompt_keys[example.prompt] = (digits, longest)
        keys.append(prompt_keys[example.prompt])

    return tuple(
        AdditionStage(
            max_digits=threshold[0],
            max_carry_chain=threshold[1],
            examples=tuple(
                example for example, key in zip(examples, keys) if key <= threshold
            ),
        )
        for threshold in sorted(set(keys))
    )


@dataclass(frozen=True)
class ArithmeticData:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    sequence_length: int
    completion_length: int
    thinking_steps: tuple[int, ...]


def generate_data(
    *, max_operand: int = 19, operations: str = "+-*", seed: int = 0,
    completion_length: int = 32, dataset_size: int = 120,
    thinking_steps: tuple[int, ...] = DEFAULT_THINKING_STEPS,
) -> ArithmeticData:
    """Balance operator/operand-length groups, then randomly split 80/10/10.

    Operands include zero and max_operand. Subtraction may be negative.
    Commuted expressions are distinct prompts and may occur in different splits.
    Sample dataset_size unique equations without enumerating the full space.
    Balance (operator, left digits, right digits) quotas within one example,
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
    if not operations or len(set(operations)) != len(operations):
        raise ValueError("operations must be nonempty and contain no duplicates")
    if any(op not in OPERATIONS for op in operations):
        raise ValueError("supported operations are +, -, and *")
    thinking_steps = tuple(thinking_steps)
    if (
        not thinking_steps
        or any(steps < 0 for steps in thinking_steps)
        or any(left >= right for left, right in zip(thinking_steps, thinking_steps[1:]))
    ):
        raise ValueError("thinking_steps must be nonempty, nonnegative, and strictly increasing")
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-operand", type=int, default=19)
    parser.add_argument("--operations", default="+-*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=120)
    parser.add_argument("--completion-length", type=int, default=32)
    parser.add_argument(
        "--thinking-steps", type=int, nargs="+", default=DEFAULT_THINKING_STEPS,
        help="Increasing counts of ~ tokens per equation; variants do not count toward dataset-size",
    )
    parser.add_argument(
        "--sort", choices=("dataset", "equation", "operand-size"), default="dataset",
        help="Viewer order within each split: sampled order, left/operator/right, or max/min operand",
    )
    options = vars(parser.parse_args())
    sort_order = options.pop("sort")
    data = generate_data(**options)
    operator_order = {op: index for index, op in enumerate(OPERATIONS)}

    def sort_key(example):
        op = next(op for op in OPERATIONS if op in example.prompt)
        left, right = map(int, example.prompt[:-1].split(op))
        equation = (left, operator_order[op], right)
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
