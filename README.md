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
        - Mean squared error (MSE): continuous regression
        - Cross-entropy/negative log-likelihood: mutually-exclusive multiclass classification
4. Gradient descent
    - Stochastic gradient descent (SGD): a compute-efficient version of gradient that only uses a **minibatch** (subset) of training data to compute the gradient.
        - A side effect of SGD is that it helps escape local minima because of its randomness
5. Backpropagation
    - An efficient algorithm for computing gradients

## Feedforward networks

- References
    - https://www.3blue1brown.com/lessons/neural-networks
    - https://www.3blue1brown.com/lessons/neural-network-analysis/
- Feedforward networks are also called **multilayer perceptrons (MLPs)**.
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
- Create a `conda`/`mamba`/`micromamba` (use the one you have installed) environment for 20.A04:
    ```bash
    micromamba create -n 20.A04 python=3.12
    ```
- Activate the environment:
    ```bash
    micromamba activate 20.A04
    ```
- Install the dependencies:
    ```bash
    pip install jax equinox einops numpy optax trackio tqdm
    ```
- Run the training script:
    ```bash
    python train.py
    ```
- In another terminal, activate the environment and then start the [Trackio](https://github.com/gradio-app/trackio) viewer:
    ```bash
    micromamba activate 20.A04
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

## Convolutional neural networks (CNNs)

`mnist/train_cnn.py` follows the same five-section structure as `train.py`.

Its `CNN(eqx.Module)` uses Equinox `Conv2d`, `MaxPool2d`, and `Linear` layers: two SAME-padded convolutions with ReLU and 2×2 max pooling, followed by a ReLU dense layer and ten raw class logits.

Metrics are logged to the `mnist-cnn` Trackio project.

```bash
python train_cnn.py --help
python train_cnn.py --channels 8 16 --kernel-size 3 --hidden-size 16 \
    --learning-rate 1.0 --batch-size 100 --epochs 10 --seed 0
```

The values above are the defaults. `--channels` takes the output channel count for each convolution; `--kernel-size` controls both square kernels; `--hidden-size` controls the dense hidden layer.

### References

- CNNs
    - How blurs and filters work (Computerphile): https://www.youtube.com/watch?v=C_zFhWdM4ic
    - Sobel edge operator (Computerphile): https://www.youtube.com/watch?v=uihBwtPIBxM
    - Convolutional neural networks (Computerphile): https://www.youtube.com/watch?v=py5byOOHZM8
- Categorical cross-entropy
    - Entropy (3Blue1Brown): https://www.youtube.com/watch?v=l6DKRf-fAAM
    - Cross-entropy (understanding categorical cross-entropy loss; 3Blue1Brown): https://www.youtube.com/watch?v=GlYgs6v2YfU

## Attention and transformers

- 3Blue1Brown
    - DL5 — Transformers: https://www.youtube.com/watch?v=wjZofJX0v4M
    - DL6 — Attention in transformers: https://www.youtube.com/watch?v=eMlx5fFNoYc