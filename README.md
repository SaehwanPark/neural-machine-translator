# Neural Machine Translator (JAX/Flax)

A minimal neural machine translation (NMT) project implemented in **JAX/Flax**, with SentencePiece tokenization and a small CLI for preparing artifacts, training, and decoding. The codebase is **language-agnostic**: it works for any source/target language pair as long as your parallel data follows the expected naming convention.

Defaults: Mandarine Chinese to American English (**zh → en**). ZH/EN language datasets can be found [here](https://web.stanford.edu/class/cs224n/assignments_w25/a3.zip).

Inspired by [the Stanford CS224n Winter 2025 NMT assignment (Torch-based)](https://web.stanford.edu/class/cs224n/assignments_w25/a3.pdf).

---

## Distinction points

- **JAX/Flax instead of PyTorch**: the model and training loop are written in Flax + Optax, with explicit PRNG handling and JAX arrays.
- **Language-agnostic**: language selection is controlled by `--src-lang` / `--tgt-lang` and file extensions (e.g., `train.de`, `train.en`).
- **Brief feature summary**:
  - SentencePiece train/load per language
  - biLSTM encoder + attention + LSTMCell decoder (teacher forcing)
  - masked cross-entropy loss, dev perplexity
  - beam search decoding
  - idiomatic Python `logging` everywhere

---

## Compute note

Training NMT models can be compute-intensive. For reasonable training times, run on **GPU/TPU** (or other accelerators) with a suitable JAX install. CPU-only runs may be very slow.

---

## Data layout

Set `DATA_DIR` (via `.env`) to a directory containing parallel files named by language abbreviation:

- `train.<src>` / `train.<tgt>`
- `dev.<src>` / `dev.<tgt>`
- `test.<src>` / `test.<tgt>`

Example (default):
- `train.zh`, `train.en`, `dev.zh`, `dev.en`, `test.zh`, `test.en`

`.env`:
```bash
DATA_DIR=/home/saehwan/data/machine_translation
````

---

## Install (uv)

```bash
uv sync
```

---

## Artifacts layout

Artifacts are stored per language pair:

```
artifacts/<src>-<tgt>/
  spm.<src>.model
  spm.<tgt>.model
  vocab.json
  meta.json
  checkpoints/
```

---

## CLI usage

### 1) Prepare SentencePiece + vocab

```bash
uv run nmt_cli.py prepare
# or
uv run nmt_cli.py prepare --src-lang de --tgt-lang en
```

### 2) Train

```bash
uv run nmt_cli.py train
# or
uv run nmt_cli.py train --src-lang de --tgt-lang en
```

### 3) Decode (beam search)

```bash
uv run nmt_cli.py decode \
  --src-file "$DATA_DIR/test.zh" \
  --out-file outputs/test.pred.en
```

For another pair:

```bash
uv run nmt_cli.py decode \
  --src-lang de --tgt-lang en \
  --src-file "$DATA_DIR/test.de" \
  --out-file outputs/test.pred.en
```

Common decode knobs:

* `--beam-size`
* `--max-len`

---

## Logging

All logs use Python `logging`:

```bash
uv run nmt_cli.py train --log-level INFO
uv run nmt_cli.py train --log-level DEBUG --log-file outputs/train.log
uv run nmt_cli.py train --log-use-tqdm-handler
```

---

## Tests

```bash
uv run pytest
```
