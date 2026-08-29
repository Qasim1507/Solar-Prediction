---
tags: [deployment]
type: note
---

# Verification loop

`verify.py` — fetches what actually happened and grades the forecast.

## Why it matters more than the forecast

**The verification loop is what caught [[Train-serve skew]].** Across eight logged days the
old system's forecasts barely moved — at t+3h the forecast varied by ±7.7 W/m² while
reality varied by ±85.5 — and what variation existed was **negatively correlated** with the
truth (−0.66 / −0.74 / −0.49).

That is invisible in MAE. The live MAE of 79.4 looked *better* than the offline 97.

> [!quote] How to frame it in the defence
> Not "my deployment worked". Instead: *"I deployed it, the verification loop caught that
> the live forecasts had no skill, and I traced it to five concrete train/serve
> divergences."* Very few master's projects contain a train/serve skew analysis at all.

## What it logs now

Per row: forecast, actual, error, CI hit — **plus** `clearsky_wm2`, `kt_forecast`,
`kt_actual`, `kt_bias`, and a [[Smart persistence]] reference.

## The report

`python3 verify.py --report` → five sections. See [[kt bias diagnostic]].

> [!caution] "measured" is a misnomer
> Actuals come from Open-Meteo — **the same product used as the training target**. It is a
> consistency check, not independent ground truth. See
> [[ISSUE-D5 Reanalysis labelled measured]].
