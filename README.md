# 20.A04-DeepLearning

## Shared Python environment

The root `pyproject.toml` declares dependencies for both MNIST and arithmetic
experiments. From the repository root:

```bash
uv sync
uv run mnist/train.py
uv run math/experiment.py
```

See [arithmetic experiments](#recurrent-neural-networks-rnns-lstms-grus) below.

## Math foundations
- [3Blue1Brown's Linear Algebra series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    - Focus on chapters 3, 4, and 5

## Neural networks, cost functions, gradient descent, and backpropagation

### Notes

1. The "purpose" of deep learning
    - Learning from data
2. Neural networks and architectures
    - **Universal approximation theorem**. Even a single hidden layer feedforward network, given enough width (number of neurons), can represent any continuous function on a compact domain (i.e. any mapping from inputs to outputs).
3. Cost/loss functions
    - Because we are learning a function to map from inputs to outputs, we need a way to inform the network how well or poorly it is performing. This is the purpose of the **cost/loss function**.
    - Typical loss functions used are:
        - **Mean squared error (MSE)**, for continuous regression
        - **Categorical cross-entropy**/negative log-likelihood, for mutually-exclusive multiclass classification
            - Entropy (3Blue1Brown): https://www.youtube.com/watch?v=l6DKRf-fAAM
            - Cross-entropy (understanding categorical cross-entropy loss; 3Blue1Brown): https://www.youtube.com/watch?v=GlYgs6v2YfU
4. Gradient descent
    - Stochastic gradient descent (SGD): a compute-efficient version of gradient that only uses a **minibatch** (subset) of training data to compute the gradient.
        - A side effect of SGD is that it helps escape local minima because of its randomness
5. Backpropagation
    - An efficient algorithm for computing gradients

### References
- https://www.3blue1brown.com/lessons/neural-networks
- https://www.3blue1brown.com/lessons/gradient-descent
- https://www.3blue1brown.com/lessons/neural-network-analysis/
- https://www.3blue1brown.com/lessons/backpropagation/
- https://www.3blue1brown.com/lessons/backpropagation-calculus/


## Setup

- If you don't have Anaconda/`conda`/`mamba` installed on your computer (WSL/macOS/Linux), then install Micromamba by running the following command in your terminal:
    ```bash
    # for Linux users
    "${SHELL}" <(curl -L micro.mamba.pm)

    # for macOS users
    brew install micromamba
    ```
- Additionall, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/#homebrew) if you don't have it already:
    ```bash
    # for Linux users
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # for macOS users
    brew install uv
    ```
- Install the dependencies:
    ```bash
    uv sync
    ```

## Feedforward networks

### Background
- What is a neural network?
    - https://www.3blue1brown.com/lessons/neural-networks
    - https://www.3blue1brown.com/lessons/neural-network-analysis/
- Feedforward networks are also called **multilayer perceptrons (MLPs)**.
- Train, validation, and test sets
  - MNIST
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Image dimensions: 28 × 28 pixels (784 pixels)
    - Classes: 10 (digits 0–9)

### Experiment

- Run the training script:
    ```bash
    uv run python train.py
    ```
- In another terminal, `cd` into this repository, and then start the [Trackio](https://github.com/gradio-app/trackio) viewer:
    ```bash
    uv run trackio show
    ```
- Now, you can play with different hyperparameters of the model and see how it affects performance:
    ```bash
    # show available options
    uv run python train.py --help

    # change number of epochs
    uv run python train.py --epochs 20

    # change hidden size; you are training on CPU, so this number can't be too high
    uv run python train.py --hidden-size 32
    ```
- You can also visualize the dataset:
    ```bash
    uv run python visualize_data.py --split train --index 0 --count 10
    ```

## Convolutional neural networks (CNNs)

### Background

- What is a kernel convolution?
    - How blurs and filters work (Computerphile): https://www.youtube.com/watch?v=C_zFhWdM4ic
    - Sobel edge operator (Computerphile): https://www.youtube.com/watch?v=uihBwtPIBxM
- What is a convolutional neural network?
    - Convolutional neural networks (Computerphile): https://www.youtube.com/watch?v=py5byOOHZM8

### Experiment

`mnist/train_cnn.py` trains a CNN to recognize the MNIST dataset of handwritten digits.

The model has two convolution layers, followed by a max pooling layer, one hidden linear layer, and finally the output layer.

Metrics are logged to the `mnist-cnn` Trackio project.

```bash
uv run python train_cnn.py --help
uv run python train_cnn.py --channels 8 16 --kernel-size 3 --hidden-size 16 \
    --learning-rate 1.0 --batch-size 100 --epochs 10 --seed 0
```

The values above are the defaults. `--channels` takes the output channel count for each convolution; `--kernel-size` controls both square kernels; `--hidden-size` controls the dense hidden layer.


## Recurrent neural networks: RNNs, LSTMs, GRUs

`math/` compares a vanilla tanh RNN and an LSTM, both supporting stacked layers,
on arithmetic completion. From the repository root:

```bash
uv run math/data.py --max-operand 19 --dataset-size 120 --seed 0
uv run math/experiment.py --name vanilla-rnn-seed0
uv run trackio show
```

The generator pages each equation once, grouped by split, with thinking lengths
listed at the top (`$PAGER` when set). Redirecting output disables interactive paging.
Use `--sort equation` for numeric left operand, operator (`+`, `-`, `*`), then
right operand order; `--sort operand-size` orders by the larger operand, then the
smaller, ascending. Sorting stays within each split; default `--sort dataset`
preserves the sampled order and never changes training data.

Initial training run:

```bash
uv run math/experiment.py \
    --max-operand 999 \
    --dataset-size 10000 \
    --hidden-size 128 \
    --batch-size 128 \
    --epochs 100 \
    --learning-rate 0.001 \
    --seed 0
```

Each epoch shows a tqdm batch progress bar with ETA and running training loss.

Choose `--model rnn` (default) or `--model lstm`; both accept `--hidden-size`
and `--num-layers` (positive integer, default 1). All recurrent layers use the
same hidden size. Layer 1 consumes token embeddings; each higher layer consumes
the current output of the layer below. Only the top layer feeds the
readout. Hidden state (and LSTM cell state) is independent per layer and resets
for every equation. The Python constructors also accept `num_layers=N`.

Artifacts record the layer count. Single-layer initialization and serialized
array order are preserved; older artifacts without `num_layers` default to 1.

Enable `--skip-connections` (Python: `skip_connections=True`) for residual
connections around layers 2 through N: each layer forwards its input plus its
new hidden output. The recurrent hidden state and LSTM cell state are not
modified by the residual addition. There is no skip around layer 1, so LSTM
embedding size may differ from hidden size; with one layer the flag is a no-op.
Skips add no parameters and are disabled by default. Artifacts save the flag;
older artifacts without it load with skips disabled.

Both models support optional AdamW regularization with `--weight-decay 0.01`.
The default is `0` (no decay, equivalent to the previous Adam optimizer); values
must be finite and nonnegative. Decay applies to all trainable parameters,
including biases and LSTM embeddings, separately from gradient clipping.
Training and evaluation losses remain masked cross-entropy without an added
regularization term. The decay setting is recorded in run and artifact config.

LSTM-only options include `--embedding-size` (defaults to hidden size) and
two mutually exclusive forget-bias initializers:

- `--forget-bias B`: constant initial bias (default 1).
- `--chrono-max-timescale T`: independently sample each unit's forget bias as
  `log(U(1, T-1))`, with finite `T >= 2`. Choose `T` for the expected sequence
  memory horizon; `T=2` gives zero forget biases. This overrides the default
  constant initializer and is also available as `LSTM(..., chrono_max_timescale=T)`.

Both initializers apply independently to every LSTM layer and leave the forget
bias trainable. Chrono initialization changes only the forget gate, not the input
gate; other parameter initialization
is unchanged for the same seed. Artifacts record `T`, and older fixed-bias
artifacts remain loadable.

Use `--model lstm --help` to see its options:

```bash
uv run math/experiment.py --model lstm --max-operand 999 --dataset-size 10000 \
    --hidden-size 128 --num-layers 3 --embedding-size 64 \
    --chrono-max-timescale 32 --batch-size 128
```

- `--dataset-size` counts unique equations across all splits (default 120);
  sampling balances operand digit lengths. Operands include 0 and `--max-operand`.
- `--thinking-steps 0 1 2 4 8` (default on both CLIs) adds five `~`-prefix variants
  per equation without consuming `--dataset-size`: 120 equations become 600 rows.
  Variants stay in the same split. Use `--thinking-steps 0` for the no-thinking baseline.
- `--operations` selects from `+`, `-`, `*`; `--completion-length` caps generation
  (default 32). Use `--help` for all options and defaults. CPU is sufficient.
- Prompt `12+7=` → completion `~~~~:19$` with four thinking steps. Training learns
  `~`, delimiter, answer, and EOS; evaluation loss scores only answer/EOS across
  variants. Exact accuracy decodes each unique equation once and ignores content
  before the final answer. Trackio project: `arithmetic`.
  After each epoch, both models log `train/exact_accuracy` to Trackio and print
  it alongside training loss. This uses the epoch-end model and the same greedy
  exact-answer scoring as validation, adding one decoding pass over unique
  training equations. Training loss remains the token-weighted average recorded
  during optimizer updates.
  Training, validation, and test also log
  `<split>/exact_accuracy/operand_lengths/<left_digits>x<right_digits>` using
  the same predictions, with no extra decoding. For example, `7+12` belongs to
  `1x2`, while `12+7` belongs to `2x1`; zero counts as one digit. Each bucket
  combines all selected operators and counts unique equations, not thinking
  variants. Buckets absent from a split are omitted, not reported as zero.
  Overall exact accuracy remains equation-weighted, not an average of buckets.
- Final weights (`model.eqx`) and run metadata (`config.json`) are saved under
  `artifacts/math/<trackio-run-name>/` (override the parent with `--output-dir`).
  Trackio generates the name unless `--name` is provided. Existing directories
  are not overwritten. The runner calls `model.save_artifact(config, output_dir)`;
  reload with `RNN.load_artifact(path)` or `LSTM.load_artifact(path)`. Optimizer
  state is not saved; existing RNN artifacts remain compatible.
- Sampling, metric, and model-interface details live in the docstrings in
  [`math/data.py`](math/data.py) and [`math/experiment.py`](math/experiment.py).

Try a saved model interactively (use the run path printed after training):

```bash
uv run math/repl.py artifacts/math/your-trackio-run-name
```

The REPL detects the saved architecture automatically. Enter `12+7` or `3-9=`.
It shows the raw completion, extracted answer, expected answer, and exact-match
result. `:help` shows input syntax; `:quit`, Ctrl-D, or Ctrl-C exits.

LSTM behavioral checks: `uv run python -m unittest discover -s math -p test_lstm_model.py`.

### Addition curriculum

Both models support `--curriculum addition`. This defaults `--operations` to `+`
and rejects other operators. Without a curriculum, the existing `+-*` default
and full-training-split schedule are unchanged.

```bash
uv run math/experiment.py --model lstm --curriculum addition \
    --max-operand 999 --dataset-size 10000 --thinking-steps 0 \
    --num-layers 3 --skip-connections --hidden-size 128 --embedding-size 64 \
    --curriculum-stage-epochs 5 --epochs 100 --seed 0
```

The fixed training split is ranked by `(maximum operand digits, longest carry
chain)`. A carry chain is a consecutive run of columns producing an outgoing
carry, including a final overflow: `2+3` has no carry, `9+1` has a chain of one,
and `99+1` has a chain of two. Stages progress through observed difficulty pairs
in that order. Each stage retains **all earlier examples** for rehearsal,
including their thinking variants. Thus a two-digit/no-carry stage still
includes one-digit additions with carries. Empty difficulty groups are skipped;
the final stage contains the entire original training split.

`--curriculum-stage-epochs` defaults to 5. Each stage receives that many epochs;
any remaining epochs continue on the final stage. A run must provide at least
`number_of_stages * curriculum_stage_epochs` epochs, otherwise it fails before
training. Stages do not reset parameters or optimizer state. Sequence length,
generation budget, seeds, validation/test splits, and targets stay fixed.
There is no new scratchpad target or vocabulary: this is an ordering curriculum,
not an explicit implementation of addition inside the network.

Trackio records `curriculum/stage`, `curriculum/max_digits`,
`curriculum/max_carry_chain`, `curriculum/train_examples`, and
`curriculum/train_equations`. Stage thresholds and sizes are also saved in run
and artifact metadata. The carry threshold applies to the current maximum
digit length; all shorter lengths remain included.

Under curriculum training, `train/loss` covers only the active stage's examples,
so its distribution changes at stage transitions. `train/exact_accuracy` and
its operand-length breakdowns still cover the **full fixed training split**.
Validation is unchanged and test evaluation still happens only after training.

This ordering targets digit addition before longer carry propagation, but
does not guarantee algorithmic generalization. The existing splits measure
held-out equations within the configured operand range; they do not establish
generalization to numbers longer than those seen in training.


## Attention and transformers

- 3Blue1Brown
    - DL5 — Transformers: https://www.youtube.com/watch?v=wjZofJX0v4M
    - DL6 — Attention in transformers: https://www.youtube.com/watch?v=eMlx5fFNoYc
