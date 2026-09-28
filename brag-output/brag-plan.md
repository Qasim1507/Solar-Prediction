# Brag Plan: Singapore Solar Nowcast (v3)

## What is this app?
A 1.2M-parameter model that reads three Himawari-8 satellite frames ten minutes
apart plus 24 hours of weather, and forecasts Singapore's solar irradiance one,
two and three hours ahead — with calibrated uncertainty bands, not just a number.

## The angle
Most ML brag videos quote the number and stop. This project's distinguishing
feature is the opposite: **every claim arrives with its own caveat attached.**
The model beats LightGBM — by 2.84 W/m², at p = 0.0402, which is marginal, and
the video says so on screen.

That honesty *is* the flex. Anyone can post a leaderboard. Showing the
p-value next to it, and calling it marginal, is the part that's rare.

No web UI exists — the product is a command line. So the CLI is the hero shot,
in mono type, exactly as it renders.

## Hook (first 2-3 seconds)
A slow sky-to-data transition: a Himawari-style gradient resolving into the
question the whole project exists to answer —

> **"Where will the sun be in three hours?"**

No logo, no product name yet. The question earns the next fifteen seconds.

## Key moments (the middle)
- **The CLI answering it.** `python predict_v3.py` resolving into three real
  forecast rows — 662.3, 534.2, 377.9 W/m² — each with its 90% interval and its
  clear-sky index. Actual output, typed in mono.
- **What's doing the work.** 3 satellite frames · 10 minutes apart · 24h of
  weather · 14 features → **1,236,359 trainable parameters.** The restraint is
  the point: a frozen encoder, deliberately small for ~5k samples.
- **The leaderboard.** Four bars animating up on held-out data (n=1015):
  v3 ensemble **63.38**, LightGBM **66.23**, Random Forest 66.34,
  smart persistence 86.01.

## Outro / punchline
The caveat, stated plainly and held long enough to read:

> **Δ −2.84 W/m² · p = 0.0402**
> *Marginally significant. We'd rather say so.*

Then the name. No triumphant swell — the music steps down instead of up.

## User flow worth showing
Entry → key action → result, all inside one terminal:
1. `python predict_v3.py` — the single command
2. Live fetch: Himawari frame + Open-Meteo weather, input check passing ±3σ
3. Three horizons with uncertainty bands appear, one row at a time

## Tone
- Preset: **polished**
- Creative direction: *quiet engineering result, stated precisely*
- Interpretation: Restraint throughout. No hype cuts, no zoom punches, no
  exclamation marks. Motion is slow and deliberate; type settles rather than
  slams. The video should feel like a well-set research figure that happens to
  move. Every number on screen is real and traceable to a committed file.

## Format: landscape — 1920x1080
## Duration: 18s

## Visual identity (from the project)
No CSS exists — this is a Python pipeline. Palette derived from the problem
domain (night sky → solar noon) and from what the CLI already prints.
- Background: `#0B1020` (deep pre-dawn navy)
- Accent: `#FFB627` (solar amber — the forecast line, the winning bar)
- Secondary: `#4CC9F0` (sky cyan — uncertainty bands, the 90% interval)
- Text: `#E8ECF4`
- Muted: `#7C89A3` (captions, units, the caveat line)
- Display font: **Space Grotesk**
- Body/data font: **JetBrains Mono** — the product is a CLI; mono is authentic,
  not a style choice
- Strongest visual element: the three-row forecast table with its interval
  brackets, and the four-bar MAE comparison

## Share copy (draft)
1.2M parameters, three Himawari frames and 24h of weather → Singapore solar
irradiance 1–3h out. Beats LightGBM by 2.84 W/m² on held-out data. p = 0.0402,
which is marginal — saying so is the whole point.

## Audio direction
- Role: sparse professional bed with restrained motion-matched accents
- Music: `happy-beats-business-moves-vol-9` (114.84 BPM, warm and unhurried —
  the least "launch-y" of the bundled set)
- Music treatment: fade in over 0.0–1.2s, sit low under the CLI scene so typing
  reads, lift very slightly under the leaderboard, then **step down** for the
  caveat line rather than resolving upward. Fade out 16.5–18.0s.
- Music cue guidance: cue preset read from
  `assets/music/cues/happy-beats-business-moves-vol-9-...music-cues.md`.
  Beat grid ≈0.525s spacing. Useful anchors: **2.12s** (hook → CLI),
  **6.34s** (CLI → internals), **10.54s** (internals → leaderboard),
  **14.76s** (leaderboard → caveat). Align the three forecast rows to beats at
  **3.70 / 4.23 / 4.75s** and the four bars to **11.06 / 11.60 / 12.12 / 12.65s**.
  Cue timing is guidance only — readability wins any conflict.
- Audio-reactive treatment: **subtle**. A faint amber glow on the winning bar
  may track music energy. Nothing pulses, nothing strobes.
- SFX posture: **sparse**, motion-matched. Keyboard for the typed command only;
  one soft interface tick per forecast row; one low impact on the final name.
  Nothing on the caveat line — silence there is deliberate.
- Audio-coupled moments: typed command (keyboard), three forecast rows arriving
  one by one (ticks), four bars rising (beat-aligned), name reveal (single low hit)
- Restraint rule: **no riser, no whoosh, no build into the outro.** The caveat
  must land in near-silence. If a cue and readability conflict, readability wins.

## Storyboard

### Scene 1 — The question — 2.5s
Deep navy field. A soft amber-to-cyan gradient drifts upward like a horizon at
dawn, suggesting a satellite view without faking one. Centered, settling in:
**"Where will the sun be in three hours?"** (Space Grotesk, large, `#E8ECF4`).
Held ~1.4s settled — a nine-word line needs it.
Sequential/interaction: none — single line settles, gradient drifts behind it.
Audio intent: quiet arrival; establish calm, not anticipation.
Audio-coupled idea: none.
Music: warm bed fading in from silence.
Transition mood: soft → Scene 2

### Scene 2 — The answer, in the terminal — 5.0s
Cut to a mono terminal card on the same navy. Line types in:
`$ python predict_v3.py` (JetBrains Mono, `#E8ECF4`).
Beneath it, dim `#7C89A3`: `✓ all inputs within ±3σ of training`.
Then three rows arrive one at a time, each holding ~0.9s:
```
t+1h  15:00    662.3 W/m²   [555.0 – 769.6]   k_t 0.783
t+2h  16:00    534.2 W/m²   [429.1 – 639.3]   k_t 0.775
t+3h  17:00    377.9 W/m²   [300.8 – 455.0]   k_t 0.794
```
GHI values amber; interval brackets cyan; labels muted.
Sequential/interaction: yes — command types, then three rows appear one by one,
beat-aligned at ~3.70 / 4.23 / 4.75s.
Audio intent: competence and calm; the machine simply answering.
Audio-coupled idea: keyboard SFX on the typed command; one soft interface tick
per row.
Music: bed sits low so typing reads clearly.
Transition mood: clean → Scene 3

### Scene 3 — What's doing the work — 4.0s
Terminal recedes. Four short mono labels arrive left to right, each holding
~0.8s: **3 satellite frames** · **10 min apart** · **24h weather** · **14 features**.
They converge into one large amber figure: **1,236,359** with a muted
`trainable parameters` beneath, and smaller still: `encoder frozen`.
Sequential/interaction: yes — four labels one by one, then the count-up resolves.
Audio intent: assembly; small parts becoming one thing.
Audio-coupled idea: counter ticks during the count-up, settling on the final digit.
Music: unchanged, steady.
Transition mood: clean → Scene 4

### Scene 4 — Held out — 4.5s
Four horizontal bars grow from the left, labelled in mono, MAE in W/m²
(lower is better, stated in a muted caption so the direction is unambiguous):
```
v3 ensemble        63.38   ← amber, longest-lasting
LightGBM           66.23
Random Forest      66.34
smart persistence  86.01
```
Header, muted: `held-out test · n = 1015 · never used for training or selection`.
Bars rise beat-aligned at ~11.06 / 11.60 / 12.12 / 12.65s; the v3 bar settles
last and holds an amber glow.
Sequential/interaction: yes — four bars arrive one by one.
Audio intent: quiet accumulation, not triumph.
Audio-coupled idea: beat-aligned bar reveals; subtle amber glow tracking energy.
Music: lifts very slightly.
Transition mood: clean → Scene 5

### Scene 5 — The caveat, then the name — 2.0s
Bars fade. Centered, amber: **Δ −2.84 W/m² · p = 0.0402**.
Below in muted `#7C89A3`, held ~1.0s: *Marginally significant. We'd rather say so.*
Music steps down. Then the name settles in Space Grotesk:
**Singapore Solar Nowcast** with `himawari-8 · era5 · 1-3h ahead` beneath.
Sequential/interaction: yes — figure, then caveat line, then name.
Audio intent: honesty landing in near-silence; deliberately anticlimactic.
Audio-coupled idea: one low impact on the name only. Nothing under the caveat.
Music: steps down, then fades out 16.5–18.0s.
Transition mood: soft → end

---

**Scene durations:** 2.5 + 5.0 + 4.0 + 4.5 + 2.0 = **18.0s** ✓ (within 15–25s)

## Accuracy note
Every figure on screen is real and traceable:
- forecast rows → `forecast_latest_v3.json` (issued 14:00 SGT 2026-09-21)
- MAE table + Δ + p-value → `analysis/v3_results_test.json`
- parameter count → `train.py` startup log, commit `2f9fc74`

Nothing is rounded in the flattering direction. The p-value is shown, not hidden.
