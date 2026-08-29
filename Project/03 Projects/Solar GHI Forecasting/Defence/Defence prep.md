---
tags: [defence, moc]
type: moc
---

# Defence prep

> [!tip] Rehearse in this order
> [[60-second answer]] → [[Three hard questions]] → [[Q and A bank]] → skim
> [[Limitations]] → check [[Open issues]] is honest.

## The frame

Lead with the **negative result**. A negative result you can explain mechanistically is
stronger than a positive one you cannot. You have the mechanism: [[Overfitting]] beginning
at the exact epoch the CNN unfreezes, [[Gate collapse]] with a measured α range, and a
monotone [[Ablation results]] ordering.

## What is genuinely strong

1. **[[Baseline results]]** — +35/+45/+51% [[Forecast skill score]] vs [[Smart persistence]]. A real positive result.
2. **[[Gate collapse]]** — diagnosed, not just observed. The flat-loss explanation is the best analysis in the project.
3. **[[Calibration results]]** — coverage *with* widths *and* a residual test.
4. **[[Train-serve skew]]** — five divergences found by your own verification loop. Almost no master's project has this.
5. **[[kt bias diagnostic]]** — you built the tool that shows 88% of live error is one constant offset.

## What to soften

- ~~"live MAE 79.4"~~ → it was a low-variance sample, anti-correlated with truth
- ~~"removing the image branch improves by 12.9"~~ → "in a single run" until [[ISSUE-E1 Single seed]] closes

## The golden rule

> [!important]
> **Ask "how many test samples?" of every table you put on a slide.** Every claim needs a
> sample size; every comparison must be on the same rows.
