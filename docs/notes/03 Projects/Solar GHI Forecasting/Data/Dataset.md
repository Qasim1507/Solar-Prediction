---
tags: [data]
type: note
---

# Dataset

`data/combined_dataset.csv`

| | |
| --- | --- |
| Rows | **9,590** |
| Period | 2024-01-01 → 2026-08-16 |
| Hours kept | **08:00–17:00 SGT** (10 rows/day) |
| Rows with a satellite image | **7,659** |
| Last row with an image | 2026-06-07 |
| GHI mean ± sd (train) | 476.9 ± 258.4 W/m² |
| Split | 70 / 15 / 15, **strictly chronological** |

## Why daylight-only

Night GHI ≈ 0, trivially predictable. Keeping it would inflate every accuracy metric and let
the loss be dominated by samples where the answer is "zero".

> [!important] Consequence — say "steps", not "hours"
> A 24-**step** window spans about **2.4 calendar days**, not 24 hours. `sin_hour`/`cos_hour`
> tell the model where in the day each step sits.

## Two live problems

> [!danger] [[ISSUE-TEST18 Test set collapsed to 18 samples]]
> Weather runs to August 2026, images stop in June → the image-less rows all landed in the
> test split. **Deep models were scored on 18 samples over 2 days.**

> [!bug] [[ISSUE-C2 Overnight gap]]
> Targets are built by positional `shift(-h)`, so 10 / 20 / 30% of t+1/2/3h targets are
> actually *next morning*.
