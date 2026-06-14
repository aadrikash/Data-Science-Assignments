# Can a flat list of numbers recognize a cat?

---

I built this to understand something that confused me for a while — why do we even need convolutional layers? Why can't we just feed images into a regular neural network and call it a day? Turns out the answer is deeply satisfying once you see it in action on real data.

This notebook pits two architectures against each other on the CIFAR-10 dataset: a standard ANN that treats images like spreadsheet rows, and a CNN that actually *looks* at images the way images deserve to be looked at. The difference in performance is not subtle.

---

## What's in here

```
CIFAR10_ANN_CNN_Assignment.ipynb   ← the whole thing, run top to bottom
README.md                          ← you're here
```

---

## The dataset

CIFAR-10. 60,000 tiny color images (32×32 pixels) split across 10 classes:

> airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

50,000 for training. 10,000 held back for testing. Each image is 3,072 numbers (32 × 32 × 3 RGB channels) that somehow encode whether you're looking at a frog or a truck. The challenge is teaching a model to figure out which.

---

## The experiment

### Round 1 — ANN (the naive approach)

Flatten every image into a 3,072-number vector. Feed it through Dense layers. Hope for the best.

```
Input (3072) → Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.2) → Dense(10)
```

What ANN *doesn't* know: pixel (15, 15) and pixel (15, 16) are neighbors. After flattening, they're just two arbitrary positions in a list. All spatial context — gone. The model has no idea that groups of pixels form shapes, and shapes form objects.

**Result: ~52% accuracy.** Technically learning. Practically, not great.

---

### Round 2 — CNN (the right tool for the job)

Three convolutional blocks, each one learning progressively richer features.

```
Conv2D(32) → BatchNorm → MaxPool      ← finds edges and color patches
Conv2D(64) → BatchNorm → MaxPool      ← combines those into textures  
Conv2D(128) → BatchNorm               ← starts recognizing object parts
Flatten → Dense(128) → Dropout(0.4) → Dense(10)
```

The filter count deliberately scales up: 32 → 64 → 128. Early layers need fewer detectors because they're only looking for simple primitives. Later layers need more because the combinations they encode are exponentially more numerous.

`padding='same'` on every Conv2D keeps spatial dimensions stable through convolution. MaxPooling does the actual shrinking (32×32 → 16×16 → 8×8) — intentionally, progressively.

**Result: ~72% accuracy.** A 20-point jump for images. Architecture matters.

---

### Round 3 — Training strategies 

Three additional experiments exploring what happens *after* you choose your architecture:

**Data augmentation** — RandomFlip, RandomRotation(0.1), RandomZoom(0.1) baked directly into the model as a preprocessing layer. The model sees a slightly different version of each image every epoch, making it much harder to memorize and much better at generalizing. Placed inside `model.Sequential` so it activates during `fit()` and quietly turns itself off during `evaluate()`.

**Extended training (20 epochs)** — Both ANN and CNN run for twice as long. ANN plateaus early. CNN keeps improving. The performance gap actually widens.

**EarlyStopping** — Max 30 epochs but monitors `val_loss` with `patience=5` and `restore_best_weights=True`. In practice it stops earlier than 30 and hands back the weights from the single best epoch, not the last one. Avoids that classic mistake of training past the model's peak.

---

## Results at a glance

| Model | Epochs | Test Accuracy |
|---|---|---|
| ANN — baseline | 10 | ~52% |
| CNN — baseline | 10 | ~72% |
| CNN — augmented | 10 | ~70% * |
| ANN — deeper layout | 20 | ~56% |
| CNN — filter scaling | 20 | ~75% |
| CNN — EarlyStopping | ≤30 | ~74% |

*Augmented CNN looks slightly lower at 10 epochs because augmented data is genuinely harder to learn from. Run both for 30 epochs and it wins.*

---

## How to run it

Open in Google Colab, switch runtime to GPU (Runtime → Change runtime type → T4 GPU), then run all cells top to bottom. The whole thing takes about 15–20 minutes.

```
Runtime → Run all
```

That's it. No pip installs, no external downloads, no config files. CIFAR-10 loads directly from `tf.keras.datasets`.

---

## What I actually learned building this

The gap between ANN and CNN isn't about size or depth. It's about inductive bias — the assumptions baked into the architecture before training even starts. ANN assumes every input feature is independent. CNN assumes neighboring pixels are related. For images, the second assumption is always true. That's why CNN wins on vision tasks even when ANN has more parameters.

BatchNorm was the most underrated addition. Without it, training was noticeably more unstable and required more careful learning rate tuning. With it, just use Adam at 0.001 and it pretty much works.

The EarlyStopping experiment taught me that more epochs ≠ better model. The best weights often appear around epoch 12–15 even when you give the model 30 chances. Without `restore_best_weights=True`, you'd return the epoch-30 model which has quietly started overfitting.

---

## Stack

- Python 3.10
- TensorFlow 2.x / Keras
- NumPy, Pandas, Matplotlib
- Google Colab (free tier, GPU runtime)

---


