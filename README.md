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
    - Gabor filters (shape and frequency): https://en.wikipedia.org/wiki/Gabor_filter
- What is a convolutional neural network?
    - Convolutional neural networks (Computerphile): https://www.youtube.com/watch?v=py5byOOHZM8
- Transformer circuits for CNNs: https://distill.pub/2020/circuits/zoom-in/

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

### Background

- Most AI has amnesia. Here's the fix (Artem Kirsanov): https://www.youtube.com/watch?v=PAoe7mmmvp0
- RNN architectures: https://inductive-research.github.io/blog/arch

### Experiment

`math/` compares a vanilla tanh RNN and an LSTM, both supporting stacked layers,
on arithmetic completion. From the repository root:

```bash
uv run math/data.py --max-operand 19 --dataset-size 120 --seed 0
uv run math/experiment.py --name vanilla-rnn-seed0
uv run trackio show
```

In `math/experiment.py`, `create_model` handles initialization or artifact loading,
`train_epoch` performs one pass over the training split, and `run_experiment`
coordinates evaluation, logging, and saving. `main` connects model setup, data
generation, and the run. The REPL shares its decoding and answer-scoring helpers.

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

To train from an existing model, pass its artifact directory (containing both
`config.json` and `model.eqx`).

```bash
uv run math/experiment.py --init-from artifacts/math/previous-run \
    --epochs 50 --learning-rate 0.0001 --name continued-run
```

Try a saved model interactively (use the run path printed after training):

```bash
uv run math/repl.py artifacts/math/your-trackio-run-name
```

The REPL detects the saved architecture automatically. Enter `12+7` or `3-9=`.
It shows the raw completion, extracted answer, expected answer, and exact-match
result. `:help` shows input syntax; `:quit`, Ctrl-D, or Ctrl-C exits.

### Addition-only training

Both models train on the full training split every epoch. The default operations
are `+-*`; pass `--operations +` to train on addition only.

```bash
uv run math/experiment.py --model lstm --operations + \
    --max-operand 999 --dataset-size 10000 --thinking-steps 0 \
    --num-layers 3 --skip-connections --hidden-size 128 --embedding-size 64 \
    --epochs 100 --seed 0
```


## Attention and transformers

- 3Blue1Brown
    - DL5 — Transformers: https://www.youtube.com/watch?v=wjZofJX0v4M
    - DL6 — Attention in transformers: https://www.youtube.com/watch?v=eMlx5fFNoYc
