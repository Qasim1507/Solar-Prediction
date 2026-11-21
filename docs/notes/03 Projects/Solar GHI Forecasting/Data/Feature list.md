---
tags: [data]
type: note
---

# The eleven tabular features

`TABULAR_COLS` in `model.py`. Each is z-scored using **training-split statistics only**,
persisted to `train_stats.json`.

| # | Feature | Why it's there |
| --- | --- | --- |
| 1 | `clearsky_ratio` | The [[Clear-sky index]] — cloud attenuation itself. Detrends the diurnal cycle. |
| 2 | `cloud_cover` | ERA5 total cloud fraction, %. Direct attenuation proxy. |
| 3 | `temperature_2m` | Correlates ~0.86 with GHI — largely a proxy for accumulated insolation. |
| 4 | `rain` | Non-zero rain implies deep cloud, strong attenuation. |
| 5 | `wind_speed_10m` | Surface proxy for advection speed of the cloud field. |
| 6 | `relative_humidity_2m` | High RH precedes convective cloud formation. |
| 7–8 | `sin_hour`, `cos_hour` | Cyclical encoding — 23:00 and 00:00 must be adjacent. |
| 9–10 | `sin_month`, `cos_month` | Season, without a Dec/Jan discontinuity. |
| 11 | `ghi_lag1` | Previous hour's GHI — the persistence signal, made explicit. |

The **tabular baselines get 14 features** — these plus `ghi_lag2`, `ghi_lag3` and raw
`ghi_clearsky`. So [[Baseline results]] are achieved on a handicap that *favours* the
baselines. Worth saying out loud.

The `no-lag` ablation drops #11 to measure how much of the model is just persistence.
