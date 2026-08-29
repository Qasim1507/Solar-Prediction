---
tags: [concept, architecture]
type: note
---

# Cross-attention

Attention answers: *"given what I'm looking for, which parts of this should I pay attention
to?"*

$$H_a = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

Here:
- **Query** = `H_t`, the temporal summary from the [[Temporal branch]] — 1 token
- **Keys / Values** = the 196 image patches from the [[Image branch]]
- 8 heads × 32 dims each, so different heads can specialise

Output: `H_a`, a 256-dim summary of the sky **as seen through the lens of the current
weather state**.

> [!question] "Why is the query length 1? Isn't that wasting the mechanism?"
> It reduces attention to a **learned pooling** over the 196 patches, weighted by where the
> weather sequence currently sits. Deliberate — you want one sky summary conditioned on the
> current state. A longer query (one token per horizon) would let each horizon attend
> differently, and is reasonable future work.
