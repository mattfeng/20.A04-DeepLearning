# math: experiments with RNNs and Transformers

`math/` compares a vanilla tanh RNN, a chrono-initialized LSTM, and a decoder-only Transformer on a simple addition completion task (e.g. `123+456=`).

From the repository root:

```bash
uv run math/data.py --data-config 'max-operand=19 dataset-size=120 seed=0'
uv run math/experiment.py --data-config 'max-operand=19 dataset-size=120 seed=0'
uv run trackio show
```

Each prompt is an equation such as `12+7=`. The target completion contains optional `~` thinking tokens followed by `:answer$`, for example `~~:19$`. Equations are balanced by operand digit lengths and split roughly 80/10/10 into training, validation, and test sets before expanding thinking-step variants.

The experiment prompts separately for model, data, and training/run settings. Press Enter to keep the displayed default, or type a replacement. Use `yes`/`no` for booleans, comma-separated integers for thinking steps (for example `0,1,2,4`), and `none` to clear an optional artifact path or run name. Ctrl-D or Ctrl-C cancels configuration before training starts.

Three optional flags supply prompt defaults using space-separated `key=value` pairs: `--model-config` for the model, `--data-config` for data generation, and `--experiment-config` for training and run settings. Quote values containing spaces inside the config string, for example `--experiment-config 'name="my run" epochs=50'`.

Model options are listed separately below so you can copy the settings for the architecture you are testing. See [Transformer model options](#transformer-model-options) under [Attention and transformers](#attention-and-transformers) for the Transformer.

## Shared experiment options

These settings apply to RNN, LSTM, and Transformer runs alike. Pass data settings in `--data-config` and training/run settings in `--experiment-config`, separately from `--model-config`.

**Data settings (`--data-config`)**

| Setting | Valid values | Default | Description |
| --- | --- | --- | --- |
| `max-operand` | Integer ≥ `1` | `19` | Largest operand; operands range from `0` through this value. |
| `dataset-size` | Integer from `10` through `(max-operand + 1)²` | `120` | Number of unique addition equations before thinking-step expansion. |
| `completion-length` | Integer ≥ `1`, large enough for every generated completion | `32` | Must fit the longest thinking prefix, `:`, answer, and `$`; the generator reports the required minimum if too small. |
| `thinking-steps` | Nonempty comma-separated, strictly increasing nonnegative integers | `0,1,2,4,8,16` | For example: `0`, `0,1,2,4`, `2,4,8`. |
| `seed` | Integer | `0` | Random seed for equation sampling and train/validation/test splits. Independent of the training/run seed. |

**Training and run settings (`--experiment-config`)**

| Setting | Valid values | Default | Description |
| --- | --- | --- | --- |
| `resume` | `yes`, `no` (also `true`, `false`; case-insensitive) | `no` | Prompted only when `init-from` is set. Use `yes` to restore the checkpoint's dataset, optimizer state, training settings, and completed epoch, or `no` to load weights only with configurable data and a fresh optimizer. |
| `batch-size` | Integer ≥ `1` | `64` | Examples per batch. |
| `epochs` | Integer ≥ `1`; greater than the completed epoch when resuming | `100` | Target epoch number. Fresh or weights-only training starts at epoch `1`; full resume continues through this target rather than adding this many epochs. |
| `learning-rate` | Finite real number > `0` | `0.001` | Adam learning rate; scientific notation such as `1e-3` is accepted. |
| `weight-decay` | Finite real number ≥ `0` | `0.0` | Parameter-penalty coefficient; `0` disables the penalty. |
| `weight-decay-type` | `l1`, `l2` | `l2` | Parameter-penalty type. |
| `seed` | Integer | `0` | Random seed for model initialization and training shuffles. Does not change the dataset. |
| `project` | Nonempty string | `arithmetic` | Trackio project name. |
| `name` | `none` or a nonempty run name usable as a single directory name (not `.` or `..`) | `none` | Trackio run name and artifact subdirectory; `none` chooses a name automatically. |
| `output-dir` | Nonempty directory path | `artifacts/math` | Artifact root; existing run directories are preserved, with name collisions resolved before training. |

The dataset viewer accepts the same `--data-config` string as the experiment command. Additionally, you can control viewer ordering via the `--sort` flag. For example, to inspect equations ordered by operand size:

```bash
uv run math/data.py \
    --data-config 'max-operand=19 dataset-size=120 thinking-steps=0,1,2,4 seed=0' \
    --sort operand-size
```

## RNN model options

| Setting | Valid values | Default | Description |
| --- | --- | --- | --- |
| `model` | `rnn`, `lstm`, `transformer` | `rnn` | Use `rnn` for this model. |
| `init-from` | `none` or an existing artifact directory path | `none` | Load an RNN artifact; omit all other model settings when using a path. |
| `hidden-size` | Integer ≥ `1` | `128` | Embedding and recurrent hidden width. |
| `num-layers` | Integer ≥ `1` | `1` | Number of stacked recurrent layers. |
| `skip-connections` | `yes`, `no` (also `true`, `false`; case-insensitive) | `no` | Add each upper layer's input to its output; the first layer has no skip. |

```bash
uv run math/experiment.py \
    --model-config 'model=rnn hidden-size=128 num-layers=1 skip-connections=no' \
    --data-config 'max-operand=999 dataset-size=10000' \
    --experiment-config 'batch-size=128'
```

## LSTM model options

| Setting | Valid values | Default | Description |
| --- | --- | --- | --- |
| `model` | `rnn`, `lstm`, `transformer` | `rnn` | Use `lstm` for this model. |
| `init-from` | `none` or an existing artifact directory path | `none` | Load an LSTM artifact; omit all other model settings when using a path. |
| `hidden-size` | Integer ≥ `1` | `128` | Recurrent hidden and cell width. |
| `num-layers` | Integer ≥ `1` | `1` | Number of stacked recurrent layers. |
| `skip-connections` | `yes`, `no` (also `true`, `false`; case-insensitive) | `no` | Add each upper layer's input to its output; the first layer has no skip. |
| `embedding-size` | Integer ≥ `1` | Chosen `hidden-size` | Token embedding width, which may differ from hidden size. |
| `chrono-max-timescale` | Finite real number ≥ `2` (`none` is not allowed) | `32.0` | Maximum timescale for chrono initialization of forget-gate biases. |

```bash
uv run math/experiment.py \
    --model-config 'model=lstm hidden-size=128 num-layers=1 skip-connections=no embedding-size=128 chrono-max-timescale=32'
```

LSTM forget-gate biases use chrono initialization: `log(U[1, chrono-max-timescale - 1))`. A timescale of `2` gives zero initial forget-gate biases. Other gate biases use the linear layer's default initialization; all biases remain trainable.

For three stacked layers with residual skips, smaller embeddings, and no thinking tokens:

```bash
uv run math/experiment.py \
    --model-config 'model=lstm num-layers=3 skip-connections=yes hidden-size=128 embedding-size=64 chrono-max-timescale=32' \
    --data-config 'max-operand=999 dataset-size=10000 thinking-steps=0'
```

## Transformer options

`math/transformer_model.py` implements a small decoder-only Transformer with scaled token embeddings, sinusoidal positional encodings, pre-normalized causal multi-head self-attention, GELU feedforward layers, residual connections, and a final normalization/readout. There is no dropout or fixed positional length cap. Causal masking prevents future tokens (including right padding) from influencing earlier predictions.

#### Transformer model options

| Setting | Valid values | Default | Description |
| --- | --- | --- | --- |
| `model` | `rnn`, `lstm`, `transformer` | `rnn` | Use `transformer` for this model. |
| `init-from` | `none` or an existing artifact directory path | `none` | Load a Transformer artifact; omit all other model settings when using a path. |
| `hidden-size` | Integer ≥ `1`, divisible by `num-heads` | `128` | Token embedding and residual-stream width. |
| `num-layers` | Integer ≥ `1` | `1` | Number of stacked Transformer blocks. |
| `num-heads` | Integer ≥ `1` that divides `hidden-size` | `4` | Attention heads per block; for width `128`: `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`. |
| `feedforward-size` | Integer ≥ `1` | `4 × hidden-size` | Inner width of each block's feedforward network. |

Transformer blocks always use residual connections; there is no `skip-connections` setting for this model.

```bash
uv run math/experiment.py \
    --model-config 'model=transformer hidden-size=128 num-layers=1 num-heads=4 feedforward-size=512'
```

## Training from an artifact

To train from an existing model, pass its artifact directory (containing both `config.json` and `model.eqx`). Its architecture remains fixed. Omit architecture keys from `--model-config` when using `init-from`. With `resume=no` (the default), saved weights are restored without reinitialization, data and training/run settings remain configurable, and training starts at epoch `1` with a fresh optimizer. The vocabulary and token IDs remain unchanged for checkpoint compatibility.

```bash
uv run math/experiment.py \
    --model-config 'init-from=artifacts/math/previous-run' \
    --data-config 'max-operand=999 dataset-size=10000 seed=7' \
    --experiment-config 'resume=no epochs=50 learning-rate=0.0001 name=continued-run'
```

## Interactive evaluation

Try a saved model interactively (use the run path printed after training):

```bash
uv run math/repl.py artifacts/math/your-trackio-run-name
```

The REPL detects the saved architecture automatically. Enter `12+7` or `3+9=`; spaces and a trailing `=` are optional. Subtraction, multiplication, and negative operands are rejected. The REPL shows the raw completion, extracted answer, expected answer, and exact-match result. `:help` shows input syntax; `:quit`, Ctrl-D, or Ctrl-C exits.
