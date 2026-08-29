---
tags: [results, key-finding]
type: experiment
---

# Gate collapse — memorise this

Measured over 1,120 test samples:

| | |
| --- | --- |
| mean α | **0.4461** |
| standard deviation | 0.0096 |
| min / max | 0.4246 / 0.4773 |
| total range | **0.053** |
| design intent | 0 → 1 |

**The gate spans 5% of its available range. It is not a switch; it is a constant.**

α never leaves [0.425, 0.477] on any test sample, at any hour, under any cloud condition.

## But the sign is right

Mean α rises monotonically from **0.438** in the cloudiest decile to **0.454** in the
clearest. Clearer sky → slightly more weight on the temporal branch, exactly as designed.
**Correct direction, no usable magnitude.**

## Why — the one-sentence answer

> [!quote] Say this
> If `H_a` carries little predictive information, the loss is nearly **flat in α**, so
> gradient descent has no reason to move it anywhere. The gate didn't fail because it was
> badly designed — **there was nothing worth gating.**

## Why it's worse than concatenation

Gating is a **bottleneck**. Concatenation keeps both 256-dim vectors and lets the next layer
use whatever is useful. Gating collapses them into a single convex combination *first*,
discarding information, and only pays off if α carries real regime information. It doesn't.
So you pay the bottleneck cost for no conditioning benefit.

Related: [[Physics gate]] · [[Ablation results]] · [[Overfitting]]
