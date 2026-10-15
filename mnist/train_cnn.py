import argparse
import gzip
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import trackio
from tqdm.auto import tqdm


# --------------------------------------------------
# 1. Hyperparameters
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--learning-rate", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--channels", type=int, nargs=2, default=[8, 16],
                    help="output channels in the two convolutional layers")
parser.add_argument("--kernel-size", type=int, default=3)
parser.add_argument("--hidden-size", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

class CNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, key, channels, kernel_size, hidden_size):
        c1, c2 = channels
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(
            1, c1, kernel_size, padding="SAME", key=keys[0]
        )
        self.conv2 = eqx.nn.Conv2d(
            c1, c2, kernel_size, padding="SAME", key=keys[1]
        )
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.hidden = eqx.nn.Linear(7 * 7 * c2, hidden_size, key=keys[2])
        self.output = eqx.nn.Linear(hidden_size, 10, key=keys[3])

    def __call__(self, image):
        # Equinox layers take one channels-first image; vmap handles batches.
        x = image[jnp.newaxis, :, :]
        x = self.pool(jax.nn.relu(self.conv1(x)))
        x = self.pool(jax.nn.relu(self.conv2(x)))
        x = jax.nn.relu(self.hidden(x.reshape(-1)))
        return self.output(x)  # Raw class logits.


# --------------------------------------------------
# 3. Loss and training
# --------------------------------------------------

optimizer = optax.sgd(args.learning_rate)


def loss(model, images, labels):
    logits = jax.vmap(model)(images)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )


@eqx.filter_jit
def train_step(model, opt_state, images, labels):
    value, gradients = eqx.filter_value_and_grad(loss)(
        model, images, labels
    )

    updates, opt_state = optimizer.update(
        gradients, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)

    return model, opt_state, value


@eqx.filter_jit
def accuracy(model, images, labels):
    predictions = jnp.argmax(jax.vmap(model)(images), axis=-1)
    return jnp.mean(predictions == labels)


def evaluate(model, images, labels, batch_size):
    # Bound convolutional activation memory, including on the test set.
    total = 0.0
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        total += float(accuracy(model, batch_images, batch_labels)) * len(batch_images)
    return total / len(images)


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

model = CNN(
    jax.random.key(args.seed), args.channels, args.kernel_size, args.hidden_size
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
rng = np.random.default_rng(args.seed)

# Trackio stores the configuration and metrics locally.
trackio.init(
    project="mnist-cnn",
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

        model, opt_state, batch_loss = train_step(
            model,
            opt_state,
            train_images[batch],
            train_labels[batch],
        )

        epoch_loss += batch_loss * len(batch)

    train_loss = float(epoch_loss / len(train_images))
    val_accuracy = evaluate(model, val_images, val_labels, args.batch_size)

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
test_accuracy = evaluate(model, test_images, test_labels, args.batch_size)
trackio.log({"test_accuracy": test_accuracy})

print(f"Test accuracy: {test_accuracy:.2%}")

trackio.finish()
