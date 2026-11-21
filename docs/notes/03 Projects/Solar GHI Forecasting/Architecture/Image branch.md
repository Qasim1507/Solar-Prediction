---
tags: [architecture]
type: note
---

# Image branch

Two [[EfficientNet]]-B2 encoders, initialised from [[SwimSeg pretraining]]:

- **`global_enc`** — shared across the three time-stacked frames (t, t−1, t−2)
- **`roi_enc`** — the centre 112×112 crop of frame t, upscaled back to 224×224
  (roughly the inner 2°×2° around Singapore)

Each 224×224 image → 352 channels at 7×7 → flattened to **49 patches** of 352 dims.

$$3 \text{ global frames} \times 49 + 1 \text{ RoI} \times 49 = \mathbf{196 \text{ patches}}$$

These become the keys and values for [[Cross-attention]].

**Cost:** 4 EfficientNet forward passes per sample — this dominates everything else in the
model. Frozen for the first 20 epochs, then unfrozen at lr/10 (see [[Training setup]] and
[[Overfitting]]).

Params: 7,202,562 each = **14.4M**, i.e. 92.9% of the model.
