# Text Generation using Vanilla RNN, LSTM, and GRU

**Deep Learning | Sequence Modelling Assignment**  
Builds and compares three recurrent neural network architectures — SimpleRNN, LSTM, and GRU — to learn the structure of a custom text corpus and generate coherent word sequences.

---

## Overview

This notebook trains three sequence models on a custom AI/ML-themed paragraph corpus. Each model learns to predict the next word given a sequence of preceding words. After training, all three models generate text from a shared seed phrase, and their learning curves are compared on a single plot.

---



## Corpus

A custom 13-sentence paragraph about artificial intelligence, machine learning, and neural networks. Cleaned to lowercase, split at sentence boundaries for n-gram construction.

```
Total characters  : 1,259
Unique words      : 129
Vocabulary size   : 125  (after tokenization)
```

---

## Pipeline — Step by Step

### Step 1 — Corpus Ingestion & Cleaning
Raw paragraph is lowercased and stripped. This ensures `AI` and `ai` are treated as the same token and no empty tokens appear at string boundaries.

### Step 2 — Tokenization → N-Grams → Padding
- `Tokenizer` builds a word-to-integer vocabulary ranked by frequency
- Progressive sliding-window n-grams are built within each sentence (not across sentence boundaries)
- All sequences are left-padded with zeros to uniform length using `pad_sequences(padding='pre')`

```
N-gram sequences  : 163
Max sequence length: 18
X shape           : (163, 17)
y shape           : (163, 125)   ← one-hot encoded
```

### Step 3 — Model Architecture
All three models share the same layer structure — only the recurrent layer differs:

```
Embedding(125, 64, input_length=17)
    ↓
SimpleRNN(128)  /  LSTM(128)  /  GRU(128)
    ↓
Dense(125, activation='softmax')
```

Compiled with:
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(learning_rate=0.01)`
- Metric: `accuracy`

### Step 4 — Training
All three architectures trained for **100 epochs** with `batch_size=32` and identical optimizer configuration, ensuring a fair comparison.

```
SimpleRNN  →  Loss: 0.0400  |  Accuracy: 99.39%
LSTM       →  Loss: 0.0122  |  Accuracy: 99.39%
GRU        →  Loss: 0.0103  |  Accuracy: 99.39%
```

### Step 5 — Training Curve Plot
A single line graph overlays loss across 100 epochs for all three models. The plot is saved as `training_curves.png`.

**Key observation:** GRU converges fastest and reaches the lowest final loss. LSTM is close behind. SimpleRNN converges last and shows higher variance in early epochs.

### Step 6 — Text Generation
The `generate_text` function extends a seed phrase one word at a time:

1. Tokenize and pad the current phrase
2. Run `model.predict()` → get a 125-element probability array
3. Apply `np.argmax()` to pick the highest-probability word index
4. Reverse-lookup the index in `tokenizer.word_index` to get the word string
5. Append the word and repeat

**Seed phrase:** `"artificial intelligence is"`

```
[SimpleRNN]  artificial intelligence is transforming every aspect of modern life game playing and protein
[LSTM]       artificial intelligence is transforming every aspect of modern life life enormous promise for
[GRU]        artificial intelligence is transforming every aspect of modern life life life produce coherent
```

---

## Student Customization Tasks

| Task | Change | Result |
|---|---|---|
| Task 1 | Custom corpus | 13-sentence AI/ML paragraph |
| Task 2 | Embedding dim 10 → **64** | Richer word representations |
| Task 3 | Train GRU for **200 epochs** | Loss: 0.0098, Acc: 99.39% |
| Task 4 | Hidden units 64 → **128** | More recurrent memory capacity |
| Task 5 | Generate **10 words** per call | `generate_text(..., next_words=10)` |

**Task 3 — 200-epoch GRU output:**
```
artificial intelligence is transforming every aspect of modern life promise for science medicine
```

**Task 5 — sample output:**
```
deep learning models inspired by the human brain use layers of artificial neurons
```

---

## Dependencies

```
tensorflow >= 2.x
numpy
matplotlib
```

Install with:
```bash
pip install tensorflow numpy matplotlib
```

---

## How to Run

1. Open the notebook in Jupyter or Google Colab
2. Run all cells top to bottom — each step depends on the previous
3. `training_curves.png` is saved automatically after Step 5
4. Edit the `corpus_text` variable in Step 1 to train on your own text
5. Call `generate_text(models['GRU'], "your seed phrase", next_words=10)` to generate from any model

---

## Results Summary

| Model | Final Loss | Convergence Speed | Notes |
|---|---|---|---|
| SimpleRNN | 0.0400 | Slowest | Prone to vanishing gradient |
| LSTM | 0.0122 | Medium | Smooth, stable descent |
| GRU | 0.0103 | Fastest | Fewest parameters, best efficiency |

GRU wins on this small corpus because its simpler gating mechanism (2 gates vs LSTM's 3) trains faster without sacrificing accuracy. On larger datasets with longer sequences, LSTM's dedicated cell state may offer an advantage.

---

## Key Concepts

- **N-gram sliding window** — builds every possible prefix of each sentence as a training sample
- **Left padding** — zeros added before real tokens so the final token is always closest to the prediction point
- **Greedy decoding** — `np.argmax` always picks the single most probable next word (deterministic, same seed = same output)
- **Categorical cross-entropy** — penalizes predictions using negative log probability; punishes confident wrong answers harshly
- **Adam optimizer** — adapts learning rate per weight based on gradient history; faster than plain SGD for NLP tasks

---

