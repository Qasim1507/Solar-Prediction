---
tags: [moc, issue]
type: moc
---

# Open issues

Sorted by what to do first. Closed ones live in [[Fixed issues]].

## Blocking the defence

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-TEST18 Test set collapsed to 18 samples]] | 🔴 critical | one line + 1h GPU |
| [[ISSUE-E1 Single seed]] | 🔴 critical | ~1h GPU |

## Live-path correctness

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-B5 Lookback window night contamination]] | 🔴 critical | one line |
| [[ISSUE-B4 Gate cloud cover synthesised]] | 🟠 high | one line |
| [[ISSUE-B6 Live weather fetched but unused]] | 🟡 low | minutes |

## Data correctness — needs a rebuild + retrain

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-C1 Clear-sky time convention]] | 🟠 high | days |
| [[ISSUE-C2 Overnight gap]] | 🟠 high | days |
| [[ISSUE-A1 Dataset versioning]] | 🟠 high | hours |
| [[ISSUE-C3 Placeholder frames in training cache]] | 🟡 low | minutes |

## Evaluation gaps

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-E2 No Diebold-Mariano test]] | 🟡 medium | 20 min |
| [[ISSUE-E3 No CRPS]] | 🟡 medium | 20 min |
| [[ISSUE-E4 Single-season test set]] | 🟡 medium | free with TEST18 |

## Code hygiene

| Issue | Severity |
| --- | --- |
| [[ISSUE-D6 CNN-only ablation mislabeled]] | 🟡 medium |
| [[ISSUE-D5 Reanalysis labelled measured]] | 🟡 medium |
| [[ISSUE-D3 NaN batches uncounted]] | 🟡 medium |
| [[ISSUE-D2 future_clearsky unnormalised]] | 🟡 low |

→ [[Next actions]] · [[Limitations]]
