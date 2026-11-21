---
tags: [concept, method]
type: note
---

# Ablation study

Retrain the same model with **one component removed or changed**, holding everything else
identical — data, loss, optimiser, schedule, seed — to isolate that component's
contribution.

The variants in this project:

| Variant | Fusion rule | Question it answers |
| --- | --- | --- |
| LSTM-only | `enrich([H_t ‖ c])` | Is the image branch contributing anything? |
| CNN-only | `enrich([H_a ‖ c])` | Is the temporal branch contributing? |
| Naive concat | `enrich([H_t ‖ H_a ‖ c])` | Does the gate beat plain concatenation? |
| **Physics-gated** | `enrich([α·H_t + (1−α)·H_a ‖ c])` | The proposal |
| No-lag | drops `ghi_lag1` | How much is just persistence? |
| Small | B0 backbone, half width | Is it overfitting? |

> [!warning] Naming problem
> "CNN-only" **still runs the BiLSTM** — `H_a` is cross-attention with `H_t` as the query.
> See [[ISSUE-D6 CNN-only ablation mislabeled]].

Results → [[Ablation results]]
