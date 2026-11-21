---
tags: [architecture]
type: note
---

# Temporal branch

Reads the 24-step window of 11 features ([[Feature list]]) and produces one 256-dim vector.

1. **[[BiLSTM]]** — 2 layers, 128 hidden, bidirectional, dropout 0.1
   → (B, 24, 256), one vector per timestep
2. **Attention pooling** — a 256→128→1 scoring MLP produces one logit per timestep; softmax
   over the 24 steps gives weights summing to 1; output is the weighted sum
   → `H_t` (B, 256)

> [!tip] Why attention pooling, not the last hidden state?
> Taking only the last state forces one vector to carry 24 steps of history through a
> bottleneck. Attention lets the model **emphasise the last three hours during a ramp** and
> a longer stretch on a stable day.

**This is where the performance actually comes from.** In [[Ablation results]] the
LSTM-only variant is the *best* model.

Params: 539,648 (LSTM) + 33,025 (attention).
