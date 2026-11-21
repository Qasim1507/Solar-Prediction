---
tags: [issue, deployment]
type: issue
status: open
severity: low
code: B6
---

# B6 — Live weather fetched but unused

> [!bug] Status: **OPEN** · severity **low**

`predict.py` loads `weather_current.json` from data.gov.sg, prints it, and writes it into
`forecast_latest.json` — but it **never enters `tabular_seq`**. The window is built entirely
from the CSV plus the Open-Meteo top-up. Deck slide 32 step 2 implies otherwise.

## Fix

Either wire the station readings into the final window row, or remove the fetch and correct the architecture diagram. Don't leave it as-is — it's a claim you can't support.
