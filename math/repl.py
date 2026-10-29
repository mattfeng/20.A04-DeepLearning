"""Interactively evaluate a saved addition model using the experiment's scoring."""

import argparse
import json
from pathlib import Path
import re


HELP = "Enter a+b (optional =). :help for help; :quit or Ctrl-D to exit."


def read_equations():
    """Yield canonical prompts and operands; reject unsupported input without eval."""
    while True:
        text = input("math> ").strip()
        if not text:
            continue
        if text in {":q", ":quit", ":exit", "quit", "exit"}:
            return
        if text == ":help":
            print(HELP)
            continue
        match = re.fullmatch(r"([0-9]+)\s*\+\s*([0-9]+)\s*=?", text)
        if match is None:
            print("Use two nonnegative integer operands joined by +; e.g. 12+7=.")
            continue
        left, right = match.groups()
        try:
            left, right = int(left), int(right)
        except ValueError as error:
            print(f"Invalid operand: {error}")
            continue
        yield f"{left}+{right}=", left, right


def predict(model, config, prompt: str) -> str:
    """Greedily decode from the prompt alone, retaining EOS for exact scoring."""
    import numpy as np

    from data import PAD_ID, VOCAB, encode
    from experiment import complete

    completion_length = config["completion_length"]
    # Reuse the training shape where possible, but allow longer evaluation inputs.
    capacity = max(config["sequence_length"], len(prompt) + completion_length)
    tokens = np.full((1, capacity), PAD_ID, dtype=np.int32)
    tokens[0, :len(prompt)] = encode(prompt)
    lengths = np.asarray([len(prompt)], dtype=np.int32)
    prediction = np.asarray(complete(model, tokens, lengths, completion_length))[0]
    generated = prediction[len(prompt):len(prompt) + completion_length]
    return "".join(VOCAB[int(token)] for token in generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Run directory containing model.eqx and config.json")
    args = parser.parse_args()

    from experiment import extract_answer
    from lstm_model import LSTM
    from rnn_model import RNN
    from transformer_model import Transformer

    try:
        config = json.loads((Path(args.artifact) / "config.json").read_text())
        model_class = {
            "RNN": RNN, "LSTM": LSTM, "Transformer": Transformer,
        }.get(config["architecture"])
        if model_class is None:
            raise ValueError(f"Unsupported artifact architecture: {config['architecture']}")
        model, config = model_class.load_artifact(args.artifact)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Could not load artifact: {error}\n")
    print(f"Loaded {args.artifact}")
    print(HELP)
    try:
        for prompt, left, right in read_equations():
            expected = str(left + right)
            completion = predict(model, config, prompt)
            answer = extract_answer(completion)
            body, eos, _ = completion.partition("$")
            print(f"  completion: {body}{eos}")
            print(f"  answer: {answer if answer is not None else '<missing :answer$>'}")
            result = "correct" if answer == expected else "incorrect"
            print(f"  expected: {expected} ({result})")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
