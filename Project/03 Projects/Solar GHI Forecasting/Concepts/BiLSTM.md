---
tags: [concept, architecture]
type: note
---

# BiLSTM

A **recurrent** network that carries a memory forward step by step, deciding at each step
what to keep and what to forget (that is the "LSTM" gating). **Bidirectional** means it
reads the sequence forwards *and* backwards.

Configuration here: `nn.LSTM(11, 128, num_layers=2, bidirectional=True, dropout=0.1)`
→ 24 output vectors of 256 dims (128 × 2 directions).

> [!question] "Isn't reading backwards cheating?"
> No. **Every one of the 24 steps is already in the past** at forecast time. The backward
> pass just gives each step context from both sides of the window. No future information
> enters.

Followed by **attention pooling** rather than taking the last hidden state — see
[[Temporal branch]].
