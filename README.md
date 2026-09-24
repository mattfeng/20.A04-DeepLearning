# 20.A04-DeepLearning

## Math foundations
- [3Blue1Brown's Linear Algebra series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    - Focus on chapters 3, 4, and 5

## Neural networks, cost functions, gradient descent, and backpropagation

### References
- https://www.3blue1brown.com/lessons/neural-networks
- https://www.3blue1brown.com/lessons/gradient-descent
- https://www.3blue1brown.com/lessons/neural-network-analysis/
- https://www.3blue1brown.com/lessons/backpropagation/
- https://www.3blue1brown.com/lessons/backpropagation-calculus/

### Notes

1. The "purpose" of deep learning
    - Learning from data
2. Neural networks and architectures
    - **Universal approximation theorem**. Even a single hidden layer feedforward network, given enough width (number of neurons), can represent any continuous function on a compact domain (i.e. any mapping from inputs to outputs).
3. Cost/loss functions
    - Because we are learning a function to map from inputs to outputs, we need a way to inform the network how well or poorly it is performing. This is the purpose of the **cost/loss function**.
    - Typical loss functions used are:
        - Mean squared error (MSE):
        - Cross-entropy: categorical classification
4. Gradient descent
    - Stochastic gradient descent
5. Backpropagation
    - An efficient algorithm for computing gradients

## MNIST in JAX

- References
    - https://www.3blue1brown.com/lessons/neural-networks
    - https://www.3blue1brown.com/lessons/neural-network-analysis/
- Train, validation, and test sets
  - MNIST
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Image dimensions: 28 × 28 pixels (784 pixels)
    - Classes: 10 (digits 0–9)

### Notes

- If you don't have Anaconda/`conda`/`mamba` installed on your computer (WSL/macOS/Linux), then install Micromamba by running the following command in your terminal:
    ```bash
    "${SHELL}" <(curl -L micro.mamba.pm)
    ```
- Create a `conda`/`mamba` (use the one you have installed) environment for 20.A04:
    ```bash
    mamba create -n 20.A04 python=3.12
    ```
- Activate the environment:
    ```bash
    mamba activate 20.A04
    ```
- Install the dependencies:
    ```bash
    pip install jax einops numpy trackio
    ```
- Run the training script:
    ```bash
    python train.py
    ```
- In another terminal, activate the environment and then start the [Trackio](https://github.com/gradio-app/trackio) viewer:
    ```bash
    mamba activate 20.A04
    trackio show
    ```
- Now, you can play with different hyperparameters of the model and see how it affects performance:
    ```bash
    # show available options
    python train.py --help

    # change number of epochs
    python train.py --epochs 20

    # change hidden size; you are training on CPU, so this number can't be too high
    python train.py --hidden-size 32
    ```
- You can also visualize the dataset:
    ```bash
    python visualize_data.py --split train --index 0 --count 10
    ```

## Attention and transformers

- 3Blue1Brown
    - DL5 — Transformers: https://www.youtube.com/watch?v=wjZofJX0v4M
    - DL6 — Attention in transformers: https://www.youtube.com/watch?v=eMlx5fFNoYc
