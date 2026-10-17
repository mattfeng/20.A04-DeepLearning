# Arithmetic completion

A vanilla tanh RNN baseline for comparing causal sequence architectures.

## Run

From the repository root:

```bash
uv sync
uv run math/data.py --max-operand 19 --dataset-size 120 --seed 0
uv run math/experiment.py --name vanilla-rnn-seed0
uv run trackio show
```

Small smoke run:

```bash
uv run math/experiment.py --epochs 1 --max-operand 3 \
    --dataset-size 48 --hidden-size 16 --batch-size 8 --name smoke
```

Defaults: 100 epochs, hidden size 128, batch size 64, Adam learning rate 0.001,
gradient clipping at 1.0, seed 0. Uses the default JAX device; CPU is sufficient.
Trackio logs locally to project `arithmetic`. No checkpoints are saved.

## Data

- Operands range from 0 through `--max-operand` (default 19). Operations are
  `+`, `-`, `*`; select a subset with `--operations`, e.g. `--operations '+'`.
  Subtraction can produce negative answers.
- `--dataset-size` (default 120) is the total across all splits. It must be
  between 10 and `len(operations) * (max_operand + 1) ** 2`, inclusive.
- Sample unique equations without enumerating the full space. Balance quotas
  across `(operator, left digit length, right digit length)` groups; redistribute
  quota from exhausted groups. Zero has one digit; ranges stop at `max_operand`.
- Shuffle and split reproducibly: `floor(N / 10)` each for validation and test,
  the remainder for training (default 96/12/12). Balance applies to the combined
  dataset, not each split. Prompts never overlap, but commuted expressions may.
  Metrics describe digit-balanced interpolation, not extrapolation or uniform
  sampling over all integer pairs.

## Format and evaluation

- Prompt `12+7=` → completion `:19$`. `:` marks the final answer, `$` is EOS,
  `_` is padding. Training examples contain no thinking tokens; vocabulary entry
  `~` is retained only for embedding/readout shape compatibility.
- Training loss covers the delimiter, answer, and EOS. Validation/test loss
  covers only answer characters and EOS. Both exclude prompts/padding and are
  normalized by their supervised token counts.
- `exact_accuracy` greedily decodes from the prompt and scores only the segment
  after the last `:` before the first EOS. Intermediate content is ignored:
  `1+1=0:9:2$` scores as `2`. Missing delimiter/EOS or any final-answer mismatch
  fails.
- `--completion-length` (default 32, available on both CLIs) caps generation,
  including intermediate content. It must fit the delimiter, longest answer,
  and EOS; generation stops at EOS or the limit.
- Log untrained validation at epoch 0, training loss and validation metrics each
  epoch, then test metrics once after the fixed schedule. Training shuffles with
  `seed + epoch`; validation/test traversal is deterministic.

## Comparing architectures

`data.py` owns generation, splits, vocabulary, and encoding; `rnn_model.py` owns
the Equinox RNN; `experiment.py` owns training, decoding, evaluation, and logging.
Imports do not generate data, train, or initialize Trackio.

Reuse `generate_data` and `run_experiment(model, data, args)`. A model must be
causal, accept integer tokens `(time,)`, and return raw logits `(time, vocab)`
using the shared vocabulary. Generic decoding recomputes sequence logits per
token rather than using architecture-specific caches.

Keep seed, dataset size, operand range, operations, splits, completion limit,
training schedule, and metrics fixed; use distinct Trackio run names.
