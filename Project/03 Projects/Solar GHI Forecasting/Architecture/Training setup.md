---
tags: [training]
type: note
---

# Training setup

| Setting | Value | Reasoning |
| --- | --- | --- |
| Loss | [[Gaussian NLL]] | Reserve sizing needs an interval |
| Optimiser | AdamW | Decoupled weight decay |
| Learning rate | 3e-5 | Conventional CNN fine-tuning rate |
| Weight decay | **0.1** | Deliberately aggressive — overfitting anticipated |
| Scheduler | ReduceLROnPlateau ×0.5, patience 8 | |
| Batch size | 128 (CUDA) / 16 (CPU/MPS) | |
| Max epochs | 150, early stop patience 15–20 on val NLL | |
| Gradient clip | global norm 1.0 | |
| Mixed precision | fp16 autocast + GradScaler | |
| **CNN frozen** | epochs 1–20 | A random head backprop'ing into a pretrained backbone destroys it |
| **CNN unfrozen** | epoch 20, lr ÷ 10 | Let the head settle first |
| Seeds | torch 42, numpy 42 — **one run only** | ← [[ISSUE-E1 Single seed]] |
| Hardware | RTX 4090, ~12.7 min per model | |

## What the curves show

→ [[Overfitting]]. Divergence starts at epoch 22, right after the unfreeze.

> [!bug] Non-finite batches are skipped with a bare `continue` and **never counted**.
> fp16 + `log σ` + `1/σ²` is a fragile combination. See [[ISSUE-D3 NaN batches uncounted]].
