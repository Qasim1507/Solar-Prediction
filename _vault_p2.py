
# ══════════════════════════════════════════════ DEFENCE
F = f"{P}/Defence"

note(f"{F}/Defence prep.md", {"tags":"[defence, moc]", "type":"moc"}, """
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
""")

note(f"{F}/60-second answer.md", {"tags":"[defence, script]", "type":"note"}, """
# The 60-second answer

For *"so, tell me what you did."* Say roughly this:

> Solar panels produce power in proportion to the sunlight hitting them, and the only thing
> that meaningfully interrupts that sunlight is cloud. Grid operators have to decide hours in
> advance how much backup generation to keep running, so they need to know how much sun there
> will be one to three hours from now.

> I built a system that forecasts solar irradiance over Singapore at one, two and three hours
> ahead. It combines two very different kinds of information: a **time series** of weather at
> the site, and **satellite images** of the cloud field around Singapore, which show weather
> that hasn't arrived yet.

> The novel part is how I combine them. Most work just glues the two representations together
> and lets the network figure it out. I argued we already know from physics *when* each source
> should matter — on a clear day the local trend is enough; under broken cloud the spatial
> picture dominates. So I built a small gate that takes physical state — clear-sky index,
> cloud cover, cloud motion — and produces a single number α that decides how much to weight
> each branch.

> I tested it against three ablations of itself and against classical baselines. The headline
> result is **negative**: adding the satellite branch made forecasts worse, the gate collapsed
> to a near-constant, and gradient-boosted trees on eleven tabular features beat every deep
> variant. I think that's the most defensible finding in the project, and I can explain
> exactly why it happened.

Then stop. Let them ask.

Related: [[Problem statement]] · [[Ablation results]] · [[Gate collapse]]
""")

note(f"{F}/Three hard questions.md",
 {"tags":"[defence, critical]", "type":"note"}, """
# The three hard questions

These can actually go badly. Volunteer two of them before you're asked.

---

## 1. "How many samples is your test set?"

**Fix it before the defence** → [[ISSUE-TEST18 Test set collapsed to 18 samples]]. Then:
*"About 1,123 samples across 116 days and two monsoon seasons."*

If unfixed, say it **first, unprompted**:

> "The deep-model table is currently scored on 18 samples because my satellite archive stops
> before my weather data does. It's a split bug, I've identified it, here's the fix, and I'm
> not drawing conclusions from that table."

---

## 2. "How many random seeds did you run?"

Currently one — and **the ablation ordering already inverted between two runs**, which is
direct evidence that run-to-run variance is comparable to the claimed effect.

3 seeds × 4 variants ≈ 1 hour GPU. → [[ISSUE-E1 Single seed]]

Fallback if unrun:

> "One seed. I would not defend the ordering within the deep variants on that basis; the
> claim I'd defend is the direction — that image information consistently did not help."

---

## 3. "Your live system reports 79 W/m² MAE — better than offline. How?"

**Do not accept the compliment.**

> "That number is from a favourable low-variance sample, not better performance. Across those
> eight days the forecast varied by only about 8 W/m² at t+3h while reality varied by 85, and
> the forecast was actually *negatively* correlated with the truth at every horizon. A
> constant would have scored similarly. The verification loop is what surfaced that, and it
> led me to five concrete train/serve divergences — three of which I've fixed."

Delivered that way this becomes one of the **strongest** moments in the defence.

→ [[Train-serve skew]] · [[kt bias diagnostic]] · [[Verification loop]]
""")

note(f"{F}/Q and A bank.md", {"tags":"[defence]", "type":"note"}, """
# Q&A bank

Answer out loud *before* reading the answer.

## Concepts

**Why forecast irradiance rather than PV power?**
PV output is close to a deterministic function of irradiance — 0.99994 correlation here — so
irradiance is the physically fundamental quantity, and forecasting it makes the model
transferable to any panel configuration. It also avoids circularity: the PV column is a
simulation derived from GHI.

**Why 1–3 hours?** → [[Why 1-3 hours]]

**Isn't future clear-sky GHI leakage?**
No. It's a closed-form function of solar zenith angle, air mass and climatological turbidity —
it depends only on the ephemeris and site latitude. Nothing about future *weather* enters it.
→ [[Clear-sky GHI]]

## Data

**How do you know there's no leakage?**
Four controls: strict chronological cut, no shuffling; each split wrapped in its own Dataset
so a lookback window can't cross the boundary (5,343 rows → 5,312 samples); normalisation
statistics from the training split only, persisted to `train_stats.json`; and `pv_actual`
excluded everywhere because it correlates 0.99994 with GHI.

**Why only 08:00–17:00?** → [[Dataset]]

**Your ground truth is reanalysis — does that invalidate the results?**
It qualifies them. ERA5 is smoothed, so true point variability is understated and every error
is against a smoothed field — which likely makes the problem look *easier*. The relative
comparison between my models is unaffected since they all face the same target. For absolute
operational accuracy I'd need pyranometer data (SERIS). → [[Limitations]]

## Architecture

**Why EfficientNet-B2?** → [[EfficientNet]]
**Why pretrain on SwimSeg?** → [[SwimSeg pretraining]]
**Why a scalar gate?**
It's the most direct test of the hypothesis, and it's *interpretable* — I can plot α against
cloudiness and check whether it behaves as designed, which is exactly how I diagnosed the
collapse. A 256-dim gate is my first future-work item, but it would have **hidden** the
failure rather than exposing it. → [[Physics gate]]

**Why is the cross-attention query length 1?** → [[Cross-attention]]

## Training and evaluation

**Why Gaussian NLL rather than MSE?** → [[Gaussian NLL]]
**Your R² is 0.79, isn't that good?** → [[MAE RMSE R2]]
**Is the LightGBM gap significant?** → [[ISSUE-E2 No Diebold-Mariano test]]
**Doesn't a wide interval guarantee high PICP?** → [[PICP and interval width]]

## Results

**Your novel contribution performed worst — isn't the project a failure?**
The project asked whether satellite fusion pays off at this data scale and answered clearly:
at ~5,000 samples against 14.4M image-branch parameters, it does not. That's a real finding
with a mechanism. A negative result I can explain is more useful to the next person than a
positive result I can't. → [[Overfitting]]

**Why does the gate do worse than naive concatenation?** → [[Gate collapse]]

**What would the satellite branch need to actually help?**
More data — years, not months. A much smaller image branch. And **infrared channels** — the
current input is effectively one visible band, so it carries no cloud-top temperature and
therefore no optical depth. I'd also predict k_t rather than GHI.

## Deployment

**How would you deploy at scale?**
One satellite tile covers many sites, so compute the patch embeddings once per tile per
timestep and share them across every site inside it, letting each site's temporal query attend
to its own neighbourhood. Per-site marginal cost collapses to roughly the BiLSTM. Latency
isn't the constraint — 64 ms against an hourly cadence — the satellite's 20–30 min
publication delay is.

**Isn't the physics clamp hiding failures?** → [[Physics clamp]]
""")

# ══════════════════════════════════════════════ CODE MAP
note(f"{P}/Code map.md", {"tags":"[reference]", "type":"note"}, """
# Code map

| File | Does what | Notes |
| --- | --- | --- |
| `model.py` | **Single source of truth** for architecture + inference helpers | [[Model architecture]] |
| `solar_pv_main.ipynb` | Training, ablations, evaluation | §1 EDA · §2 baselines · §3 images · §3.5 SwimSeg · §4 RAM+flow · §5 dataset · §6 model · §7 train · §8 evaluate · §9 plots |
| `historical_data.py` | Open-Meteo ERA5 + pvlib clear-sky | [[Data sources]] |
| ~~`himawari_data.py`~~ | Historical satellite tiles | ⚠️ **deleted** — restore it |
| `combined_dataset.py` | Aligns everything into one table | [[Pipeline]] |
| `current_data.py` | Live weather + satellite, with `validate_tile()` | [[Live pipeline]] |
| `predict.py` | Live forecast → `forecast_latest.json` | [[Live pipeline]] |
| `verify.py` | Verification + [[kt bias diagnostic]] report | [[Verification loop]] |
| `daily_run.sh` | Cron entry point | ⚠️ expiry `20260816` |
| `fetch_solcast.py` | PVGIS PV reference (fetched, excluded as leaky) | |

## Artefacts

| Path | What |
| --- | --- |
| `data/combined_dataset.csv` | 9,590 rows → [[Dataset]] |
| `best_model*.pt` + `.config.json` | 6 checkpoints with architecture sidecars |
| `swimseg_encoder.pt` | [[SwimSeg pretraining]] output |
| `train_stats.json` | Normalisation statistics |
| `results/*.csv` | [[Baseline results]] · [[Ablation results]] |
| `results/plots/` | eda · training_curves · horizon_degradation · gate_analysis · forecast_vs_actual |
| `analysis/` | Per-model prediction dumps, α analysis, figure builder |
| `deck/` | `build.py` → the defence pptx |
| `verification_log.csv` | Post-fix live log |
| `verification_log_prefix.csv` | Pre-fix live log (the broken era) |

## Key functions

- `model.py: build_lookback_window` ← [[ISSUE-B5 Lookback window night contamination]]
- `model.py: compute_gate_features` ← [[ISSUE-B4 Gate cloud cover synthesised]]
- `model.py: compute_clearsky_hour_mean` ← the fix that [[ISSUE-C1 Clear-sky time convention]] needs applied upstream
- `model.py: smart_persistence_forecast` / `skill_score` → [[Forecast skill score]]
- notebook cell 19 `GHIForecastDataset` ← [[ISSUE-TEST18 Test set collapsed to 18 samples]]
""")

# ══════════════════════════════════════════════ DAILY
note(f"{P}/Daily log.md", {"tags":"[log]", "type":"moc"}, """
# Daily log

Running log for the thesis. Newest first. Full daily notes live in `02 Daily`.

## 2026-08-24
- Built this vault.

## 2026-08-21
- Re-audit of commit `56b7371`. **Found [[ISSUE-TEST18 Test set collapsed to 18 samples]]** —
  deep models scored on 18 samples, baselines on 1,437.
- Rewrote `verify.py` with the [[kt bias diagnostic]].
- First post-fix verification: k_t bias **−0.180** vs **−0.220** pre-fix. Same pathology.
  Removing that one constant takes MAE 99.0 → 12.0.

## 2026-08-20
- Fix round landed: [[ISSUE-B1 No Image placeholder]], [[ISSUE-B2 Previous frames zero-filled]],
  [[ISSUE-B3 Optical flow always zero]], [[ISSUE-D1 load_model could not load concat checkpoint]] closed.
- Full retrain; added `small` preset and no-lag ablation.

## 2026-08-19
- First audit. Live deployment found to have **negative** forecast/actual correlation.
""")

note("02 Daily/2026-08-24.md", {"tags":"[daily]", "type":"daily", "date":"2026-08-24"}, """
# 2026-08-24

## Focus
- [ ] [[ISSUE-TEST18 Test set collapsed to 18 samples]] — the one-line split fix
- [ ] [[ISSUE-B5 Lookback window night contamination]] — daylight filter

## Log
- Set up this vault.

## Notes

## Tomorrow
""")

# ══════════════════════════════════════════════ AREAS / RESOURCES
note("04 Areas/README.md", {"tags":"[meta]"}, """
# Areas

Ongoing responsibilities with **no end date** — as opposed to `03 Projects`, which have a
deadline and a finish line.

Examples you might add: `Research skills`, `Python & PyTorch`, `Writing`, `Career`.

When a project finishes, its durable knowledge often graduates into an area; the project
itself goes to `06 Archive`.
""")

note("05 Resources/README.md", {"tags":"[meta]"}, """
# Resources

Reference material that isn't tied to one project — papers, links, code snippets, course
notes.

See [[References]].
""")

note("05 Resources/References.md", {"tags":"[reference]", "type":"note"}, """
# References

## Data

- **Open-Meteo Historical Weather API** — ERA5 reanalysis, hourly. `archive-api.open-meteo.com`
- **Open-Meteo Forecast API** — `past_days` analysis values; live top-up and verification
- **Himawari-8/9 real-time imagery, NICT** — `himawari8-dl.nict.go.jp`, D531106 level 4d, tile 1_1, 550px
- **JAXA P-Tree** — `ftp.ptree.jaxa.jp`, NetCDF `albedo_03` (fallback source)
- **PVGIS 5.3, EU JRC** — PV reference series (fetched, then excluded as leaky)
- **data.gov.sg** real-time environment API — station temperature, rainfall, humidity, wind

## Software

- **timm / EfficientNet** — Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- **PyTorch** — `nn.LSTM`, `nn.MultiheadAttention` (Vaswani et al., *Attention Is All You Need*, 2017)
- **pvlib** — Ineichen–Perez clear-sky model
- **OpenCV** — Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, 2003
- **LightGBM**, **scikit-learn**

## Datasets

- **SwimSeg** (NTU) — 1,013 Singapore whole-sky images with binary cloud/sky masks, Oct 2013 – Jul 2015

## Worth reading for the viva

- Diebold & Mariano (1995), *Comparing Predictive Accuracy* — the test in [[ISSUE-E2 No Diebold-Mariano test]]
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules* — CRPS, [[ISSUE-E3 No CRPS]]
- Yang et al., *Verification of deterministic solar forecasts* — where [[Forecast skill score]] conventions come from
""")

note("06 Archive/README.md", {"tags":"[meta]"}, """
# Archive

Finished or abandoned. **Never delete — just move here.** Links from other notes keep
working, and old context is often what you need six months later.
""")

# ══════════════════════════════════════════════ TEMPLATES
T = "99 Templates"

note(f"{T}/Daily note.md", {"tags":"[daily]", "type":"daily", "date":"{{date}}"}, """
# {{date}}

## Focus
- [ ]

## Log

## Notes

## Tomorrow
""")

note(f"{T}/Concept note.md", {"tags":"[concept]", "type":"note"}, """
# {{title}}

**One-sentence definition.**

## Why it matters here

## Detail

## Related
- [[ ]]
""")

note(f"{T}/Experiment note.md",
 {"tags":"[results]", "type":"experiment", "date":"{{date}}", "n_test":"", "status":"draft"}, """
# {{title}}

## Question

## Setup

| | |
| --- | --- |
| Test set / n | |
| Split | |
| Seeds | |
| Checkpoint | |

## Results

| Model | t+1h | t+2h | t+3h |
| --- | ---: | ---: | ---: |

## Reading

## Caveats
- Sample size:
- Confounds:

## Related
- [[ ]]
""")

note(f"{T}/Issue note.md",
 {"tags":"[issue]", "type":"issue", "status":"open", "severity":"medium", "code":""}, """
# {{title}}

> [!bug] Status: **OPEN** · severity **medium**

## Symptom

## Evidence

## Root cause

## Fix

## Related
- [[ ]]
""")

note(f"{T}/Meeting note.md",
 {"tags":"[meeting]", "type":"meeting", "date":"{{date}}", "with":""}, """
# {{title}}

**Date:** {{date}} · **With:**

## Agenda

## Notes

## Decisions

## Actions
- [ ]
""")

# ══════════════════════════════════════════════ WRITE
written = skipped = 0
for rel, content in N.items():
    path = os.path.join(VAULT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and open(path, encoding="utf-8").read() == content:
        skipped += 1
        continue
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    written += 1

# Obsidian config — wikilinks, attachments, and a readable graph
cfgdir = os.path.join(VAULT, ".obsidian")
os.makedirs(cfgdir, exist_ok=True)
def merge_json(name, updates):
    p = os.path.join(cfgdir, name)
    try:
        cur = json.load(open(p, encoding="utf-8"))
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        cur = {}
    cur.update(updates)
    json.dump(cur, open(p, "w", encoding="utf-8"), indent=2)

merge_json("app.json", {
    "useMarkdownLinks": False,
    "newLinkFormat": "shortest",
    "attachmentFolderPath": "05 Resources/attachments",
    "alwaysUpdateLinks": True,
    "showUnsupportedFiles": False,
})
merge_json("appearance.json", {"accentColor": "#B8660A"})
merge_json("core-plugins.json", {
    "templates": True, "daily-notes": True, "graph": True, "backlink": True,
    "outgoing-link": True, "tag-pane": True, "outline": True,
    "page-preview": True, "search": True, "file-explorer": True,
    "global-search": True, "switcher": True, "command-palette": True,
    "bookmarks": True, "properties": True,
})
json.dump({"folder": "99 Templates"},
          open(os.path.join(cfgdir, "templates.json"), "w"), indent=2)
json.dump({"folder": "02 Daily", "template": "99 Templates/Daily note",
           "format": "YYYY-MM-DD"},
          open(os.path.join(cfgdir, "daily-notes.json"), "w"), indent=2)

# Remove Obsidian's stub welcome note now that a real Home exists
stub = os.path.join(VAULT, "Welcome.md")
if os.path.exists(stub) and "This is your new" in open(stub, encoding="utf-8").read():
    os.remove(stub)

print(f"notes written : {written}")
print(f"unchanged     : {skipped}")
print(f"total notes   : {len(N)}")
print(f"vault         : {VAULT}")
