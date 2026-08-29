---
tags: [concept, background]
type: note
---

# Problem statement

Formally: **given observations up to time _t_, estimate the conditional distribution of
[[GHI]] at t+1h, t+2h and t+3h.**

Distribution, not point estimate — a grid operator sizing reserve capacity needs to know
the uncertainty. That is why the model emits μ and σ per horizon and trains on
[[Gaussian NLL]].

## The chain of reasoning

1. **PV output ≈ irradiance.** In this dataset the simulated PV series correlates with GHI
   at r = 0.99994. So forecasting solar power reduces to forecasting irradiance.
2. **Irradiance without cloud is solved.** [[Clear-sky GHI]] depends only on solar geometry
   and is computable decades ahead.
3. **Therefore the whole problem is cloud.** This single sentence justifies the entire
   architecture — the satellite branch exists because that is what shows the cloud field.
4. **Singapore is the hard case.** Equatorial maritime tropics: deep convective cloud forms
   and dissipates on 10–60 minute scales, so [[Clear-sky index]] swings between 0.1 and 1.0
   within one afternoon.

See also [[Why 1-3 hours]].
