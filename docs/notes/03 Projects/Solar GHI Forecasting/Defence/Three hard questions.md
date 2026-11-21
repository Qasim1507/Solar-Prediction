---
tags: [defence, critical]
type: note
---

# The three hard questions

These can actually go badly. Volunteer two of them before you're asked.

---

## 1. "How many samples is your test set?"

**Fix it before the defence** → [[ISSUE-TEST18 Test set collapsed to 18 samples]]. Then:
*"About 1,123 samples across 116 days and two monsoon seasons."*

If unfixed, say it **first, unprompted**:

> "The deep-model table is currently scored on 18 samples because my satellite archive stops
> before my weather data does. It's a split bug, I've identified it, here's the fix, and I'm
> not drawing conclusions from that table."

---

## 2. "How many random seeds did you run?"

Currently one — and **the ablation ordering already inverted between two runs**, which is
direct evidence that run-to-run variance is comparable to the claimed effect.

3 seeds × 4 variants ≈ 1 hour GPU. → [[ISSUE-E1 Single seed]]

Fallback if unrun:

> "One seed. I would not defend the ordering within the deep variants on that basis; the
> claim I'd defend is the direction — that image information consistently did not help."

---

## 3. "Your live system reports 79 W/m² MAE — better than offline. How?"

**Do not accept the compliment.**

> "That number is from a favourable low-variance sample, not better performance. Across those
> eight days the forecast varied by only about 8 W/m² at t+3h while reality varied by 85, and
> the forecast was actually *negatively* correlated with the truth at every horizon. A
> constant would have scored similarly. The verification loop is what surfaced that, and it
> led me to five concrete train/serve divergences — three of which I've fixed."

Delivered that way this becomes one of the **strongest** moments in the defence.

→ [[Train-serve skew]] · [[kt bias diagnostic]] · [[Verification loop]]
