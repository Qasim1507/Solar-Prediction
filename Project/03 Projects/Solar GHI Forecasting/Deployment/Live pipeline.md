---
tags: [deployment]
type: note
---

# Live pipeline

A cron job that forecasts, then grades itself.

1. **10:00 SGT** — `daily_run.sh predict`
2. **Fetch** — `current_data.py`: current satellite tile + t−1h + t−2h, each validated for
   real content; recent hourly weather to top up the archive lag
3. **Build & run** — `predict.py`: 24-step window → `load_model` (architecture auto-detected
   from the `.config.json` sidecar) → μ, σ
4. **[[Physics clamp]]** — cap at 1.15 × preceding-hour-mean clear-sky
5. **Write** `forecast_latest.json` with a `diagnostics` block (`image_ok`,
   `outside_training_hours`, `lookback_ends`, `capped`)
6. **Later** — [[Verification loop]]

## Health checks now in place

- Satellite tile rejected if mean brightness < 0.02 or > 95% black → retries backwards in
  10-min steps, then falls back to other sources (`SATELLITE_SOURCES=nict,slider,gk2a,jaxa`)
- Warns if the image is > 2h old or the lookback data is > 6h stale

> [!bug] Two divergences still open
> [[ISSUE-B4 Gate cloud cover synthesised]] and
> [[ISSUE-B5 Lookback window night contamination]] — see [[Train-serve skew]].

> [!warning] `daily_run.sh` has a hard expiry date of `20260816` and silently exits after
> it. Change it or the cron does nothing.
