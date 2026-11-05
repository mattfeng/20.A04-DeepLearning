# mnist

## Feedforward networks

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

## Convolutional neural networks

`mnist/train_cnn.py` trains a CNN to recognize the MNIST dataset of handwritten digits.

The model has two convolution layers, followed by a max pooling layer, one hidden linear layer, and finally the output layer.

```bash
uv run python train_cnn.py --help
uv run python train_cnn.py --channels 8 16 --kernel-size 3 --hidden-size 16 \
    --learning-rate 1.0 --batch-size 100 --epochs 10 --seed 0
```

The values above are the defaults. `--channels` takes the output channel count for each convolution; `--kernel-size` controls both square kernels; `--hidden-size` controls the dense hidden layer.
