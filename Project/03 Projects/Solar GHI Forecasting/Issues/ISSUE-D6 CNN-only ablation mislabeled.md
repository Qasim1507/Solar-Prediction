---
tags: [issue, code]
type: issue
status: open
severity: medium
code: D6
---

# D6 — CNN-only ablation mislabeled

> [!bug] Status: **OPEN** · severity **medium**

`ablation == "cnn"` uses `H_a`, and `H_a = cross_attn(H_t, patches)` where **the query is the
BiLSTM output**. So the temporal branch is still trained and still drives the attention
pooling. It is not "CNN-only" — it is "image features pooled by a temporal query".

Deck slide 22 says "α forced to 0; H_t discarded". **H_t is not discarded.** An examiner will
spot this. See [[Ablation study]] · [[Cross-attention]].

## Fix

Rename the variant in code, CSVs and deck ("cross-attention only" / "image branch only") **or** add a genuine image-only variant that pools patches with a learned constant query. Renaming is the honest, cheap option.
