---
tags: [issue, evaluation]
type: issue
status: open
severity: medium
code: E2
---

# E2 — No Diebold-Mariano test

> [!bug] Status: **OPEN** · severity **medium**

No statistical test of whether the LightGBM-vs-fusion gap is significant. A paired
**Diebold–Mariano** test on the forecast errors is the standard instrument. Already conceded
in deck appendix A8.

## Fix

~10 lines of scipy on the paired errors, per horizon. ~20 minutes.
