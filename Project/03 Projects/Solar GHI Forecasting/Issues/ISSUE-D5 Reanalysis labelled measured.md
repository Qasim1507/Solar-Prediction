---
tags: [issue, code]
type: issue
status: open
severity: medium
code: D5
---

# D5 — Reanalysis labelled measured

> [!bug] Status: **OPEN** · severity **medium**

`verify.py` logs `source = "measured"` for Open-Meteo `shortwave_radiation` — **the same
product used to build the training targets**. The verification is therefore model-vs-model,
and "verified against measured irradiance" (deck slides 1, 5, 32) is too strong.
See [[Verification loop]].

## Fix

Relabel the column `openmeteo_analysis`; change deck wording to "verified against Open-Meteo analysis (the same reanalysis product used as the training target — not independent ground truth)".
