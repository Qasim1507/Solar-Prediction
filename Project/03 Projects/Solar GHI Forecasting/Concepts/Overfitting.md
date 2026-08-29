---
tags: [concept, training]
type: note
---

# Overfitting — the root cause here

The model memorises the training set instead of learning generalisable structure. Diagnosed
by training loss continuing to fall while validation loss rises.

## The evidence in this project

Training and validation NLL track each other until **epoch 22**, then diverge. Validation
bottoms around epoch 26 and climbs for 20 epochs until early stopping.

**The divergence begins immediately after the CNN unfreeze at epoch 20**, and epoch time
doubles from 10s to 20s at the same point — confirming the extra parameters went live.

## Why it was inevitable

| | |
| --- | --- |
| Training samples | ~5,300 |
| Image-branch parameters | **14,405,124** (92.9% of the model) |

Releasing 14.4M parameters onto 5.3k samples is textbook. This is the mechanism behind
[[Ablation results]] and [[Gate collapse]] — and you have a plot of it.

The untested control that would prove it: **keep the CNN frozen throughout**. ~25 min of
GPU. See [[Next actions]].
