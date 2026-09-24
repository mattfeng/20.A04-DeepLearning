import argparse
import gzip
from pathlib import Path
from urllib.request import urlretrieve

import einops
import jax
import jax.numpy as jnp
import numpy as np
import trackio
from tqdm.auto import tqdm


# --------------------------------------------------
# 1. Hyperparameters
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--learning-rate", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--hidden-size", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

def init(key, hidden_size):
    sizes = (784, hidden_size, hidden_size, 10)
    keys = jax.random.split(key, len(sizes) - 1)

    return [
        # this is LeCun normal initialization
        # cf. Xavier/Glorot normal initialization
        # cf. He/Kaiming normal initialization
        (
            jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(n_in),
            jnp.zeros(n_out),
        )
        for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:])
    ]


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def predict(params, images):
    x = einops.rearrange(
        images, "batch height width -> batch (height width)"
    )

    for weights, bias in params:
        x = sigmoid(x @ weights + bias)

    return x


# --------------------------------------------------
# 3. Loss and training
# --------------------------------------------------

def loss(params, images, labels):
    targets = jnp.eye(10)[labels]
    outputs = predict(params, images)

    return jnp.mean(jnp.sum((outputs - targets) ** 2, axis=-1))


@jax.jit
def train_step(params, images, labels, learning_rate):
    value, gradients = jax.value_and_grad(loss)(
        params, images, labels
    )

    params = jax.tree.map(
        lambda p, g: p - learning_rate * g,
        params,
        gradients,
    )

    return params, value


@jax.jit
def accuracy(params, images, labels):
    predictions = jnp.argmax(predict(params, images), axis=-1)
    return jnp.mean(predictions == labels)


# --------------------------------------------------
# 4. MNIST
# --------------------------------------------------

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def read_file(filename, header_size):
    path = Path("data") / filename

    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urlretrieve(BASE_URL + filename, path)

    data = gzip.decompress(path.read_bytes())

    return np.frombuffer(
        data, dtype=np.uint8, offset=header_size
    )


def load_mnist(split):
    prefix = "train" if split == "train" else "t10k"

    images = read_file(f"{prefix}-images-idx3-ubyte.gz", 16)
    labels = read_file(f"{prefix}-labels-idx1-ubyte.gz", 8)

    images = images.reshape(-1, 28, 28).astype(np.float32) / 255

    return images, labels


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

train_images, train_labels = load_mnist("train")
test_images, test_labels = load_mnist("test")

# Reserve 5,000 training examples for validation.
val_images = train_images[-5000:]
val_labels = train_labels[-5000:]

train_images = train_images[:-5000]
train_labels = train_labels[:-5000]

params = init(jax.random.key(args.seed), args.hidden_size)
rng = np.random.default_rng(args.seed)

# Trackio stores the configuration and metrics locally.
trackio.init(
    project="mnist-3b1b",
    config=vars(args),
)

for epoch in range(args.epochs):
    indices = rng.permutation(len(train_images))
    epoch_loss = 0.0

    for start in tqdm(
        range(0, len(indices), args.batch_size),
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        unit="batch",
    ):
        batch = indices[start:start + args.batch_size]

        params, batch_loss = train_step(
            params,
            train_images[batch],
            train_labels[batch],
            args.learning_rate,
        )

        epoch_loss += batch_loss * len(batch)

    train_loss = float(epoch_loss / len(train_images))
    val_accuracy = float(accuracy(params, val_images, val_labels))

    # One Trackio logging step per epoch.
    trackio.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation accuracy: {val_accuracy:.2%}"
    )

# Evaluate the held-out test set after training.
test_accuracy = float(accuracy(params, test_images, test_labels))
trackio.log({"test_accuracy": test_accuracy})

print(f"Test accuracy: {test_accuracy:.2%}")

trackio.finish()
