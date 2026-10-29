"""Shared parsing for space-separated CLI configuration strings."""

import shlex
from collections.abc import Mapping


def parse_config(text: str | None, defaults: Mapping[str, object], flag: str) -> dict[str, str]:
    """Parse supplied overrides, rejecting keys outside this flag's group."""
    config: dict[str, str] = {}
    if text is None:
        return config
    try:
        entries = shlex.split(text)
    except ValueError as error:
        raise ValueError(f"could not read {flag}: {error}") from error
    for entry in entries:
        key, separator, value = entry.partition("=")
        option = key.replace("-", "_")
        if not separator or option not in defaults:
            raise ValueError(f"expected a known key=value option for {flag}, got {entry!r}")
        config[option] = value
    return config


def thinking_steps(text: str) -> tuple[int, ...]:
    steps = tuple(int(step) for step in text.split(","))
    if (
        not steps
        or any(step < 0 for step in steps)
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("use comma-separated increasing nonnegative integers, e.g. 0,1,2,4")
    return steps
