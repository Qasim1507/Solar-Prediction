# -*- coding: utf-8 -*-
"""Builds the M.Eng. defence deck. Every number traces to the repository."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

FIG = 'analysis/figs/'
prs = new_deck()
N = [0]
def S(title, kicker=None, rule=ACC):
    N[0] += 1
    return slide_base(prs, title, kicker, N[0], rule)

# ══════════════════════════════════════════════════════════════ TITLE ════════
s = prs.slide_layouts[6]
t = prs.slides.add_slide(s)
t.background.fill.solid(); t.background.fill.fore_color.rgb = WHITE
rect(t, 0, 0, 0.32, 7.5, fill=NAVY)
txt(t, 1.1, 1.35, 11.3, 0.3, "MASTER'S PROJECT  ·  COMPUTER ENGINEERING", size=12,
    color=ACC, bold=True)
txt(t, 1.1, 1.85, 11.2, 1.6,
    "Physics-Gated Satellite–Weather Fusion for\nShort-Horizon Solar Irradiance Forecasting",
    size=34, color=NAVY, bold=True, spacing=1.06)
line(t, 1.1, 3.62, 5.2, 3.62, color=ACC, lw=2.5)
txt(t, 1.1, 3.85, 11.0, 0.9,
    "Hourly GHI at t+1 h, t+2 h and t+3 h for Singapore, from Himawari-8/9 imagery, "
    "ERA5 reanalysis weather and a clear-sky physical prior — trained, ablated, "
    "statistically tested and deployed as an automated daily forecast.",
    size=14.5, color=INK, spacing=1.15)
tag(t, 1.1, 5.05, 2.55, 0.34, "15.5 M-parameter model", size=10)
tag(t, 3.78, 5.05, 2.35, 0.34, "9,590 hourly rows", size=10)
tag(t, 6.26, 5.05, 2.5, 0.34, "7,659 satellite frames", size=10)
tag(t, 8.89, 5.05, 2.9, 0.34, "6 trained variants + 5 baselines", size=10)
txt(t, 1.1, 5.85, 11.0, 0.6,
    "Presented for the M.Eng. project viva  ·  21 August 2026", size=13, color=INK)
txt(t, 1.1, 6.35, 11.0, 0.6,
    "Every figure and table in this deck was regenerated from the repository during "
    "preparation; the reproduction procedure and its agreement with the committed "
    "results are documented in Appendix A4.", size=10.5, color=GREY, italic=True)
note(t, """
Open with the one-sentence framing: this project forecasts global horizontal irradiance
one to three hours ahead for Singapore by fusing geostationary satellite imagery with
reanalysis weather, under a clear-sky physical prior.

Say up front what the honest headline is, because it sets the tone for the whole defence:
the system works, it beats the standard smart-persistence reference by 20–38% depending on
horizon, but the satellite branch does not pay for itself — a weather-only sequence model
and a gradient-boosted tree both match or beat the full fusion model. The project's real
contribution is that this negative result is established carefully rather than hidden.

Also flag early that I found and diagnosed a serious evaluation defect in my own committed
results (an 18-sample test split) and that slide 10 deals with it head-on. Examiners
respect finding your own bug more than they respect a clean-looking table.
""")

# ═══════════════════════════════════════════════ 1 · PROBLEM & MOTIVATION ════
s = S("Forecasting irradiance where clouds move faster than the grid",
      "Problem & motivation")
bullets(s, 0.55, 1.42, 7.35, 5.2, [
 (0, [("The problem.  ", {'bold':True,'color':NAVY}),
      ("Predict global horizontal irradiance (GHI, W/m²) over Singapore at t+1 h, "
       "t+2 h and t+3 h from data available at issue time — the horizon that matters "
       "for unit commitment, reserve sizing and storage dispatch.", {})]),
 (0, [("Why it is hard here.  ", {'bold':True,'color':NAVY}),
      ("Singapore is equatorial and convective. In the assembled dataset the mean "
       "cloud cover is 85.6% and 54% of daylight hours record rain, yet mean GHI is "
       "still 481 W/m². Irradiance is not low — it is volatile.", {})]),
 (0, [("Why persistence is not enough.  ", {'bold':True,'color':NAVY}),
      ("Clear-sky persistence — the standard operational reference — degrades quickly "
       "with horizon because it assumes today's cloud state simply follows the sun:", {})],
     {'after':4}),
], size=13.2)
table(s, 0.72, 3.72, 6.9,
      ["Reference forecast", "t+1 h", "t+2 h", "t+3 h", "degradation"],
      [["Persistence  (GHI held constant)", "155.7", "253.9", "316.5", "+103%"],
       ["Smart persistence  (kₜ held constant)", "78.9", "117.5", "143.9", "+82%"]],
      col_w=[3.0,1.0,1.0,1.0,1.2], size=10.5,
      align=[PP_ALIGN.LEFT,PP_ALIGN.CENTER,PP_ALIGN.CENTER,PP_ALIGN.CENTER,PP_ALIGN.CENTER])
caption(s, 0.72, 4.52, 6.9,
        "MAE in W/m², on the 914-sample evaluation set used throughout this deck.")
bullets(s, 0.55, 4.86, 7.35, 1.9, [
 (0, [("The opportunity.  ", {'bold':True,'color':NAVY}),
      ("A ground station cannot see the cloud field that will arrive in two hours. "
       "A geostationary satellite can: Himawari-8/9 delivers a full disc every 10 minutes, "
       "and at typical advection speeds the clouds that will shade Singapore in two hours are "
       "already inside the ~4°×4° tile at issue time.", {})]),
 (0, [("The engineering question.  ", {'bold':True,'color':NAVY}),
      ("Does that upstream visual information actually survive the journey into a "
       "forecast — on a single site, with 6,675 daylight training samples?", {})]),
], size=13.2)
panel(s, 8.2, 1.42, 4.6, 5.2)
txt(s, 8.42, 1.62, 4.16, 0.3, "SITE AND SIGNAL", size=10.5, color=ACC, bold=True)
rows = [("Location", "Singapore, 1.3521° N 103.8198° E"),
        ("Modelled hours", "08:00–17:00 SGT (daylight only)"),
        ("Target", "GHI at t+1 h, t+2 h, t+3 h"),
        ("Mean GHI", "481.1 W/m²  (sd 259.3)"),
        ("Max GHI", "1,016 W/m²"),
        ("Mean cloud cover", "85.6 %"),
        ("Hours with rain", "54.0 %"),
        ("Record length", "2024-01-01 → 2026-08-16"),
        ("Hourly rows", "9,590"),
        ("Rows with imagery", "7,659  (79.9 %)")]
yy = 2.02
for k, v in rows:
    txt(s, 8.42, yy, 1.85, 0.28, k, size=10.5, color=GREY)
    txt(s, 10.3, yy, 2.3, 0.28, v, size=10.5, color=INK, bold=True)
    yy += 0.44
note(s, """
Main message: the forecasting problem here is a volatility problem, not a resource problem.

Numbers to have ready. Mean GHI 481 W/m2 with standard deviation 259 — that is more than
half the mean, which is why a model can score a flattering R-squared just by tracking the
diurnal cycle. That is exactly why every result in this deck is reported against smart
persistence rather than against a naive mean.

Explain smart persistence properly, because an examiner will ask. Naive persistence says
GHI at t+h equals GHI at t. Smart persistence holds the clear-sky index k_t = GHI / GHI_clearsky
constant and multiplies it by the known future clear-sky value, so it gets the sun's geometry
right for free and only assumes the cloud state is unchanged. That is a much stronger
reference: 78.9 versus 155.7 W/m2 at t+1h. Any model that cannot beat it has learned only
the diurnal cycle.

Note the degradation percentages: smart persistence loses 82% of its accuracy going from
one to three hours. That decay is the space a real forecast has to work in, and slide 12
shows the models fill more of it as the horizon grows.

Likely question: "why only 08:00 to 17:00?" Answer: the dataset builder filters to those
hours (combined_dataset.py, daylight_only), because outside them clear-sky GHI is near zero
and the clear-sky index is undefined or numerically unstable. It also means the model has
never seen a night hour — which becomes an important deployment issue on slide 17.
""")

# ═══════════════════════════════════════════ 2 · OBJECTIVES & QUESTIONS ══════
s = S("What the project set out to establish", "Objectives & research questions")
txt(s, 0.55, 1.4, 12.2, 0.3,
    "Project objective (derived from the implementation — the repository contains no "
    "written objectives document; see Appendix A4)", size=11, color=GREY, italic=True)
panel(s, 0.55, 1.72, 12.23, 0.72, fill=RGBColor(0xEC,0xF1,0xF7))
txt(s, 0.8, 1.86, 11.8, 0.5,
    "Build an end-to-end, reproducible and operationally deployed short-horizon GHI "
    "forecaster for Singapore that fuses geostationary imagery with reanalysis weather, "
    "and test honestly whether the imagery and the fusion mechanism earn their cost.",
    size=13.5, color=NAVY, bold=True, spacing=1.06)
qs = [
 ("RQ1", "Does satellite imagery improve short-horizon GHI accuracy over a weather-only "
         "sequence model trained on identical data?",
  "ablation grid, notebook cell 25  →  slides 11, 13"),
 ("RQ2", "Does a physics-conditioned gate fuse the two modalities better than naive "
         "concatenation or a fixed blend?",
  "gate module, model.py:326-329  →  slides 13, 15"),
 ("RQ3", "Do 15.5 M-parameter deep models beat gradient-boosted trees when only ~6.7 k "
         "daylight training hours exist?",
  "baseline grid, notebook cell 6  →  slides 11, 13"),
 ("RQ4", "Are the predicted Gaussian intervals calibrated well enough to be used "
         "operationally?",
  "σ head + NLL objective  →  slide 14"),
 ("RQ5", "Does an offline-validated model survive deployment into an automated live "
         "pipeline?",
  "predict.py / verify.py / cron  →  slide 17"),
]
yy = 2.72
for tagn, q, where in qs:
    tag(s, 0.55, yy+0.04, 0.72, 0.34, tagn, color=WHITE, fill=NAVY, size=11)
    txt(s, 1.42, yy, 8.0, 0.62, q, size=13.2, color=INK, spacing=1.05)
    txt(s, 9.55, yy+0.05, 3.25, 0.5, where, size=9.5, color=GREY, italic=True, spacing=1.02)
    line(s, 0.55, yy+0.72, 12.78, yy+0.72, color=LGREY, lw=0.6)
    yy += 0.86
txt(s, 0.55, 7.02-0.68, 12.2, 0.5,
    "Technical objectives that follow: a single model definition shared by training and "
    "inference (model.py), probabilistic rather than point outputs, and every comparison "
    "made against persistence baselines on identical rows.",
    size=11.5, color=GREY, italic=True, spacing=1.05)
note(s, """
Be explicit and unprompted about provenance here: the repository has no written proposal or
objectives file, so these five research questions are my reconstruction from what the code
actually tests. The ablation grid in notebook cell 25 defines RQ1 and RQ2; the baseline grid
in cell 6 defines RQ3; the Gaussian negative-log-likelihood objective and the sigma head
define RQ4; predict.py, verify.py and the cron script define RQ5. I label them as derived
rather than documented — that distinction matters and examiners notice when you blur it.

Emphasise the framing choice: RQ1 and RQ2 are written so that a negative answer is a
result. I did not set out to prove imagery helps; I set out to measure whether it does.
That is why the ablation grid includes an LSTM-only variant that never uses the image
branch at all — if the imagery contributes nothing, that variant will say so.

If asked "what would have made this a stronger experimental design?", the honest answer is
multiple random seeds per variant and a rolling-origin cross-validation instead of a single
chronological split. Both are on the limitations slide, and I can quantify why they matter:
two runs of the identical configuration in this project differed by 4.5 W/m2 on validation
MAE, which is larger than several of the gaps between variants.
""")

# ═══════════════════════════════════════ 3 · BACKGROUND & GAP ════════════════
s = S("What already existed, what I built, and where the gap is", "Background & gap analysis")
hdr = ["Component", "Status", "What it is / what I did with it"]
rows = [
 ["Himawari-8/9 imagery (NICT)", ("EXISTING", {'color':GREY}),
  "Geostationary full disc, 10-min cadence. I use the 4d/550 px tile (1,1) that covers Singapore."],
 ["Open-Meteo ERA5 archive", ("EXISTING", {'color':GREY}),
  "Hourly reanalysis weather and shortwave radiation — the GHI target and all weather features."],
 ["pvlib Ineichen clear-sky model", ("EXISTING", {'color':GREY}),
  "Deterministic clear-sky GHI from solar geometry; the denominator of the clear-sky index."],
 ["EfficientNet-B2 (Tan & Le, 2019)", ("EXISTING", {'color':GREY}),
  "ImageNet-pretrained CNN backbone, used through timm as a frozen-then-fine-tuned encoder."],
 ["SWIMSEG (Dev et al., 2017)", ("EXISTING", {'color':GREY}),
  "1,013 labelled Singapore sky/cloud images used to pretrain the encoder for cloud awareness."],
 ["Smart-persistence baseline", ("EXISTING", {'color':GREY}),
  "The standard reference in solar forecasting; implemented in model.py:679 and used everywhere."],
 ["Data assembly pipeline", ("IMPLEMENTED", {'color':TEAL}),
  "historical_data.py + fetch_solcast.py + combined_dataset.py: fetch, clear-sky, daylight "
  "filter, ±60 min image matching → 9,590-row table."],
 ["Physics-gated fusion architecture", ("IMPLEMENTED", {'color':TEAL}),
  "model.py: dual CNN branches, 8-head cross-attention over 196 patches, 4-input gate, "
  "heteroscedastic 3-horizon head."],
 ["SWIMSEG pretraining stage", ("INTEGRATED", {'color':ACC}),
  "FPN decoder + Dice/BCE + cloud-fraction auxiliary head wrapped around the B2 encoder "
  "(notebook §3.5); encoder weights carried into the forecaster."],
 ["Optical-flow gate input", ("IMPLEMENTED", {'color':TEAL}),
  "Farnebäck flow between consecutive hourly frames, averaged over the Singapore RoI, "
  "fed to the gate as an advection prior."],
 ["Ablation + baseline protocol", ("CONTRIBUTION", {'color':NAVY}),
  "Six trained variants and five baselines scored on identical rows, with skill scores, "
  "Diebold–Mariano tests, CRPS and interval width."],
 ["Live deployment + verification", ("CONTRIBUTION", {'color':NAVY}),
  "predict.py / verify.py / cron: validated satellite fetch, 3-frame series, physics cap, "
  "diagnostics written into every forecast."],
]
table(s, 0.55, 1.45, 12.23, hdr, rows, col_w=[3.0,1.35,7.9], size=10, hsize=10.5,
      row_h=0.325, head_h=0.3)
txt(s, 0.55, 6.05, 12.2, 0.9,
    "The gap this project addresses:  published satellite-fusion results are usually "
    "reported on large multi-site archives, and the fusion mechanism is rarely ablated "
    "against a weather-only model or a gradient-boosted tree on the same rows. "
    "The open question for a single equatorial site with ~6.7 k training hours is not "
    "“can a fusion network be built” — it is “does the imagery pay for its 14.4 M parameters”.",
    size=12.5, color=NAVY, spacing=1.1)
note(s, """
The purpose of this slide is to draw a hard line between what I used and what I made, before
anyone has to ask. Nothing on the EXISTING rows is claimed as my work: the satellite feed,
the reanalysis product, the clear-sky model, the CNN backbone, the segmentation dataset and
the persistence baseline all predate the project.

What is mine: the data assembly pipeline, the fusion architecture as assembled in model.py,
the way SWIMSEG pretraining is wired into the forecaster, the optical-flow gate input, the
evaluation protocol, and the live deployment.

Be careful with the word novel. The architecture is an assembly of standard components —
BiLSTM with attention pooling, a CNN encoder, multi-head cross-attention, a gating MLP, a
heteroscedastic Gaussian head. None of those are new. What is new is this specific
combination, conditioned on physical variables, evaluated with the rigour on slides 11 to 16
for a site and data regime where nobody has reported it. I say that plainly rather than
overselling it, and slide 20 states it the same way.

Likely question: "why EfficientNet-B2 and not a vision transformer?" Answer: parameter
efficiency at 224x224 with roughly seven thousand training samples. A ViT would need far
more data or heavier pretraining, and the results on slide 11 show that even B2 was already
too much capacity for this dataset.
""")

# ═══════════════════════════════════════ 4 · SYSTEM ARCHITECTURE ═════════════
s = S("From three public feeds to a verified daily forecast",
      "System architecture")
# lane labels
for ly, lab, col in [(1.45,"OFFLINE  ·  data assembly and training",NAVY),
                     (4.35,"ONLINE  ·  daily automated forecast",ACC)]:
    txt(s, 0.55, ly, 4.0, 0.24, lab, size=10, color=col, bold=True)
panel(s, 0.55, 1.72, 12.23, 2.42, fill=RGBColor(0xF7,0xF9,0xFC))
panel(s, 0.55, 4.62, 12.23, 2.05, fill=RGBColor(0xFD,0xF7,0xF2))
# offline row 1: sources
srcs = [("Open-Meteo ERA5\narchive API", "hourly weather + GHI\nhistorical_data.py"),
        ("PVGIS / pvlib\nPV simulation", "pv_actual reference\nfetch_solcast.py"),
        ("Himawari-8/9\nNICT tile server", "550 px 4d tile, 10 min\n9,186 PNG frames"),
        ("SWIMSEG\n1,013 sky images", "cloud/sky masks\nencoder pretraining")]
x0 = 0.75
for i,(a,b) in enumerate(srcs):
    box(s, x0+i*2.05, 1.9, 1.85, 0.72, a, [b], edge=NAVY, tsize=10, lsize=8)
box(s, 9.15, 1.9, 3.4, 0.72, "combined_dataset.py",
    ["merge on hourly SGT timestamp · daylight 08–17 filter",
     "±60 min image matching  →  combined_dataset.csv (9,590 × 16)"],
    edge=NAVY, tsize=10, lsize=8)
arrow(s, 8.78, 2.26, 9.12, 2.26, color=NAVY, lw=1.1)
line(s, 10.85, 2.63, 10.85, 2.78, color=NAVY, lw=1.1)
line(s, 0.95, 2.78, 10.85, 2.78, color=NAVY, lw=1.1)
arrow(s, 0.95, 2.78, 0.95, 2.93, color=NAVY, lw=1.1)
# offline row 2
steps = [("Feature engineering","k_t, sin/cos hour & month,\nghi_lag1-3, targets t+1..3h"),
         ("Image tensor cache","PNG → grayscale 224×224\n→ .npy  (7,654 frames)"),
         ("Optical flow","Farnebäck between\nconsecutive frames → (vx,vy)"),
         ("Chronological split","70 / 15 / 15, no shuffle\n6,675 / 914 / 18 imaged"),
         ("Training","Gaussian NLL, AdamW,\nAMP, early stopping")]
x0 = 0.75
for i,(a,b) in enumerate(steps):
    box(s, x0+i*2.42, 2.95, 2.2, 0.72, a, [b], edge=NAVY, tsize=10, lsize=8)
    if i < 4: arrow(s, x0+i*2.42+2.22, 3.31, x0+(i+1)*2.42-0.02, 3.31, color=NAVY)
txt(s, 0.75, 3.82, 12.0, 0.26,
    "artefacts  →  best_model*.pt (6 checkpoints + .config.json sidecars)   ·   "
    "train_stats.json (per-column mean/std, GHI mean/std)   ·   swimseg_encoder.pt",
    size=9.5, color=NAVY, italic=True)
# online lane
on = [("current_data.py","validated tile fetch:\n3 frames t, t−1h, t−2h"),
      ("model.py helpers","extend_with_recent →\n24-step lookback window"),
      ("load_model()","config sidecar →\nexact architecture"),
      ("PhysicsGatedFusionV2","μ, σ for 3 horizons\n(normalised)"),
      ("Physics cap","1.15 × hour-mean\nclear-sky ceiling"),
      ("forecast_latest.json","μ, 90% band, clear-sky,\ndiagnostics flags")]
x0 = 0.72
for i,(a,b) in enumerate(on):
    box(s, x0+i*2.03, 4.85, 1.85, 0.72, a, [b], edge=ACC, tcolor=ACC, tsize=9.5, lsize=8)
    if i < 5: arrow(s, x0+i*2.03+1.87, 5.21, x0+(i+1)*2.03-0.02, 5.21, color=ACC)
box(s, 4.55, 5.85, 4.2, 0.6, "verify.py  (T + 3.5 h, cron)",
    ["re-fetches the three target hours and appends forecast vs outcome to verification_log.csv"],
    edge=ACC, tcolor=ACC, tsize=9.5, lsize=8)
arrow(s, 11.9, 5.6, 8.8, 6.12, color=ACC)
txt(s, 0.55, 6.62, 12.2, 0.3,
    "Single source of truth: model.py holds the architecture, the column order, the "
    "normalisation helpers and the baselines — so a checkpoint trained in the notebook "
    "loads unchanged at inference.", size=10, color=NAVY, italic=True)
note(s, """
Walk the diagram left to right along the top lane, then left to right along the bottom lane.

Top lane, offline. Three independent public feeds. Open-Meteo's ERA5 archive gives hourly
weather and the shortwave radiation channel that becomes the GHI target. PVGIS plus a pvlib
simulation gives a PV power reference column. NICT serves the Himawari tile. SWIMSEG is a
one-off dataset used only to pretrain the image encoder. combined_dataset.py merges them on
the floored hourly Singapore timestamp, filters to 08:00-17:00, and matches each row to the
nearest satellite frame within 60 minutes — in practice the median offset is zero minutes
because the archive is on the hour.

Then the notebook does feature engineering, converts every PNG to a grayscale 224x224 tensor
cached as .npy, precomputes Farneback optical flow between consecutive frames, splits
chronologically 70/15/15 with no shuffling, and trains.

Bottom lane, online. This runs unattended from cron. current_data.py fetches three frames —
t, t minus one hour, t minus two hours — and validates each one for brightness and structure
so a dead tile is rejected rather than fed to the network. The lookback window is topped up
from the live Open-Meteo API because the archive lags about five days. load_model reads the
config sidecar so the exact architecture is reconstructed. The physics cap limits the output
to 1.15 times the hour-mean clear-sky value. Everything, including diagnostic flags, is
written to forecast_latest.json, and verify.py re-fetches the target hours three and a half
hours later to score it.

The point to stress: model.py is imported by the notebook, by predict.py and by verify.py.
There is one architecture definition, one tabular column order and one set of baselines in
the project. That is what makes train-serve drift detectable at all, and slide 17 shows the
divergences that remain despite it.
""")

# ═══════════════════════════════════════ 5 · DATASET & PIPELINE ══════════════
s = S("Data lifecycle — and the coverage gap that shaped the results",
      "Dataset & data pipeline")
pic(s, 0.55, 1.4, FIG+'f1_coverage.png', w=7.55)
caption(s, 0.55, 4.66, 7.55,
        "Figure 1 — monthly hourly rows (light) vs rows with a matched satellite frame (dark), "
        "with the split boundaries.")
table(s, 0.55, 4.96, 7.55,
      ["Split", "Rows", "With image", "Date range", "Deep samples"],
      [["train", "6,701", "6,701  (100%)", "2024-01-01 → 2025-11-02", "6,675"],
       ["validation", "1,436", "930  (64.7%)", "2025-11-02 → 2026-03-25", ("914", {'color':TEAL})],
       ["test", "1,437", "18  (1.3%)", "2026-03-25 → 2026-08-16", ("18", {'color':RED})]],
      col_w=[1.2,0.85,1.35,2.4,1.25], size=10, row_h=0.28,
      align=[PP_ALIGN.LEFT,PP_ALIGN.CENTER,PP_ALIGN.CENTER,PP_ALIGN.LEFT,PP_ALIGN.CENTER])
caption(s, 0.55, 6.16, 7.55,
        "Deep samples = rows with an image at t plus a complete 24-step window and 3-hour "
        "horizon (valid_indices).")
bullets(s, 0.55, 6.42, 7.6, 0.6, [
 (0, [("Consequence.  ", {'bold':True,'color':RED}),
      ("The weather table reached August 2026 while the image archive stalled on 2026-06-07, "
       "so the image-less tail landed almost entirely in the test split — see slide 10.",
       {})], {'size':11}),
], size=11.5)
panel(s, 8.3, 1.4, 4.48, 5.62)
txt(s, 8.52, 1.58, 4.1, 0.26, "LIFECYCLE, STEP BY STEP", size=10.5, color=ACC, bold=True)
steps = [
 ("1  Fetch", "Open-Meteo archive: temperature, RH, rain, wind, cloud cover, "
              "shortwave radiation (→ GHI), DNI, diffuse."),
 ("2  Physics", "pvlib Ineichen clear-sky GHI per timestamp; rows with "
                "GHI_clear ≤ 0 dropped (night)."),
 ("3  Filter", "08:00–17:00 SGT retained — 10 rows per day."),
 ("4  Match", "Nearest Himawari frame within ±60 min (median offset 0 min); "
              "1,931 rows find none."),
 ("5  Engineer", "k_t = GHI / GHI_clear; sin/cos of hour and month; ghi_lag1-3; "
                 "targets by positional shift(−1,−2,−3)."),
 ("6  Encode", "PNG → grayscale → 224×224 → float32 [0,1] → 3 identical channels → "
               "ImageNet z-score."),
 ("7  Motion", "Farnebäck flow at 64×64 between consecutive frames; mean (vx, vy) over "
               "the central 32 px RoI."),
 ("8  Normalise", "Per-column z-scores and GHI mean/std computed on train only, "
                  "persisted to train_stats.json."),
 ("9  Batch", "24-step windows, batch 128, chronological order preserved — no shuffling "
              "across the split boundaries."),
]
yy = 1.92
for k, v in steps:
    txt(s, 8.52, yy, 0.95, 0.24, k, size=10, color=NAVY, bold=True)
    txt(s, 9.5, yy, 3.12, 0.56, v, size=9.5, color=INK, spacing=1.0)
    yy += 0.56
txt(s, 8.52, 6.55, 4.1, 0.4,
    "11 tabular features × 24 steps  +  4 image tensors  +  3 clear-sky values  +  4 gate inputs",
    size=9.5, color=NAVY, bold=True, spacing=1.05)
note(s, """
Two things to land here: what the pipeline does, and the coverage asymmetry.

The pipeline first. Everything is keyed on the floored hourly Singapore timestamp. The
daylight filter comes from the clear-sky model — rows where clear-sky GHI is zero are night
and are dropped, then an explicit 08:00-17:00 filter leaves exactly ten rows per day. Images
are matched by nearest neighbour within sixty minutes; because both the weather archive and
the Himawari cache are on the hour, the median matching offset is zero minutes and the worst
case is sixty.

Normalisation statistics are computed on the training split only and written to
train_stats.json. That file is what makes deployment possible: predict.py loads exactly the
same mean and standard deviation vector, so a live lookback window is scaled the way the
training data was. I verified that the statistics I recompute from the CSV match the
committed train_stats.json exactly — that check is in Appendix A4.

Now the asymmetry, which is the important half of this slide. Look at the figure: the light
bars continue to August 2026 but the dark bars stop in June, and the imagery coverage
collapses right after the validation boundary. The dataset was extended with new weather
rows without backfilling the satellite archive. Because the split is positional over all
rows and the image requirement is applied afterwards, the test split kept only eighteen
imaged samples. Slide 10 is entirely about that.

If asked why the images stopped: the historical bulk collector for satellite frames is no
longer in the repository, so the weather feed kept updating and the image feed did not.
That is recorded in the discrepancy log in Appendix A4.
""")

# ═══════════════════════════════════════ 6 · MODEL ARCHITECTURE ══════════════
s = S("PhysicsGatedFusionModelV2 — 15.5 M parameters, four inputs, six outputs",
      "Model architecture")
# input column
ins = [("tabular_seq","(B, 24, 11)","24 daylight hours × 11 features"),
       ("multi_frame","(B, 9, 224, 224)","frames t, t−1h, t−2h stacked"),
       ("roi_image","(B, 3, 224, 224)","centre 112² crop, bilinear ↑224"),
       ("future_clearsky","(B, 3)","pvlib GHI at t+1..t+3 (known)"),
       ("gate_features","(B, 4)","k_t, cloud/100, flow vx, flow vy")]
txt(s, 0.55, 1.42, 2.5, 0.24, "INPUTS", size=10, color=ACC, bold=True)
yy = 1.70
for a,b,c in ins:
    box(s, 0.55, yy, 2.6, 0.62, a, [b, c], edge=LGREY, tcolor=NAVY, tsize=9.5, lsize=8,
        align=PP_ALIGN.LEFT)
    yy += 0.68
txt(s, 3.45, 1.42, 5.2, 0.24, "ENCODERS AND FUSION", size=10, color=ACC, bold=True)
box(s, 3.45, 1.72, 2.55, 1.12, "BiLSTM encoder",
    ["2 layers × 128 hidden, bidirectional, dropout 0.1",
     "→ tanh attention pooling over 24 steps",
     "(B,24,11) → (B,256)      572,673 params"], edge=NAVY, tsize=10.5, lsize=8.2)
box(s, 3.45, 3.0, 2.55, 1.3, "Dual EfficientNet-B2",
    ["global_enc: 3 frames, shared weights",
     "roi_enc: RoI crop",
     "last stage only → (B,352,7,7) each",
     "4 × 49 = 196 patches × 352 ch",
     "2 × 7,202,562 = 14.4 M  (92.9%)"], edge=NAVY, tsize=10.5, lsize=8.2)
box(s, 6.25, 1.72, 2.55, 1.12, "8-head cross-attention",
    ["query = BiLSTM state (1 token)",
     "keys/values = 196 image patches",
     "→ H_a (B,256)     419,328 params"], edge=NAVY, tsize=10.5, lsize=8.2)
box(s, 6.25, 3.0, 2.55, 1.3, "Physics gate",
    ["Linear(4→32) → ReLU → Linear(32→1) → σ",
     "α ∈ (0,1) weights temporal vs image",
     "193 params — the whole gate",
     "fused = α·H_t + (1−α)·H_a"], edge=ACC, tcolor=ACC, tsize=10.5, lsize=8.2)
arrow(s, 3.2, 2.28, 3.43, 2.28); arrow(s, 3.2, 3.6, 3.43, 3.6)
arrow(s, 6.02, 2.28, 6.23, 2.28); arrow(s, 6.02, 3.6, 6.23, 3.6)
arrow(s, 5.0, 2.86, 5.0, 2.98, color=NAVY)
txt(s, 9.0, 1.42, 3.8, 0.24, "HEAD AND OUTPUT", size=10, color=ACC, bold=True)
box(s, 9.0, 1.72, 3.78, 0.95, "enrich",
    ["concat[fused (256), future_clearsky (3)] = 259",
     "Linear(259→256) → ReLU → Dropout 0.15   ·   66,560 params"],
    edge=NAVY, tsize=10.5, lsize=8.2)
box(s, 9.0, 2.82, 3.78, 1.18, "prediction head",
    ["Linear(256→128) → LayerNorm → GELU → Drop",
     "Linear(128→64) → LayerNorm → GELU",
     "Linear(64→6)          41,926 params"], edge=NAVY, tsize=10.5, lsize=8.2)
box(s, 9.0, 4.16, 3.78, 0.85, "heteroscedastic output",
    ["μ = out[:, :3]   (3 horizons, z-scored GHI)",
     "σ = softplus(out[:, 3:]) + 1e-4  > 0"], edge=TEAL, tcolor=TEAL, tsize=10.5, lsize=8.2)
arrow(s, 8.82, 2.2, 8.98, 2.2); arrow(s, 10.9, 2.69, 10.9, 2.8, color=NAVY)
arrow(s, 10.9, 4.02, 10.9, 4.14, color=NAVY)
panel(s, 0.55, 5.15, 12.23, 1.62)
txt(s, 0.78, 5.3, 5.6, 0.26, "ABLATION SWITCHES  (single constructor argument)",
    size=10, color=ACC, bold=True)
abl = [("ablation=None", "fused = α·H_t + (1−α)·H_a", "the full physics-gated model"),
       ("ablation='lstm'", "fused = H_t", "image branch removed from the fusion path"),
       ("ablation='cnn'", "fused = H_a", "image patches pooled by the BiLSTM query*"),
       ("ablation='concat'", "fused = [H_t ; H_a]", "no gate; enrich widens to 515→256")]
yy = 5.62
for a,b,c in abl:
    txt(s, 0.78, yy, 2.05, 0.24, a, size=9.5, color=NAVY, bold=True, font="Consolas")
    txt(s, 2.95, yy, 2.5, 0.24, b, size=9.5, color=INK, font="Consolas")
    txt(s, 5.6, yy, 4.2, 0.24, c, size=9.5, color=GREY)
    yy += 0.28
txt(s, 10.0, 5.62, 2.7, 1.0,
    "* the attention query is still the BiLSTM state, so this variant is not "
    "image-only — it is renamed accordingly throughout this deck.",
    size=9, color=RED, italic=True, spacing=1.03)
note(s, """
Take this slide slowly; it is where the technical credibility is won or lost.

Left column, the four inputs. The tabular sequence is twenty-four consecutive daylight rows
of eleven features — note that is twenty-four daylight hours, roughly 2.4 calendar days, not
twenty-four clock hours. The multi-frame tensor stacks three hourly frames channel-wise into
nine channels. The RoI is the centre 112-by-112 crop of the current frame, upsampled back to
224, which zooms in on roughly the inner two degrees around Singapore. future_clearsky is
legitimate look-ahead information: clear-sky GHI depends only on solar geometry, so it is
known perfectly in advance and using it is not leakage. The gate features are the clear-sky
index, cloud cover, and the two optical-flow components.

Middle, the encoders. The BiLSTM is two layers, 128 hidden per direction, so 256-dimensional
output, pooled over the twenty-four steps by a small tanh attention module rather than by
taking the last state — that lets the model weight the informative hours. The image side
runs one EfficientNet-B2 over each of the three global frames with shared weights, and a
second B2 over the RoI crop. Each produces a 352-channel 7-by-7 map, which flattens to
forty-nine patches; four maps give 196 patch tokens.

Cross-attention: a single query token, the BiLSTM state, attends over those 196 patches with
eight heads. So the temporal state asks the image "what should I look at" and gets back a
256-dimensional summary.

The gate is the smallest and most interesting module: 193 parameters mapping four physical
scalars to a single alpha through a sigmoid. The fusion is a convex combination — alpha times
the temporal state plus one minus alpha times the attended image state. Slide 15 shows what
alpha actually learned.

Right, the head. Clear-sky values are concatenated onto the fused vector before enrich, so
the network gets the geometry explicitly. The final layer emits six numbers: three means and
three raw scale values passed through softplus to guarantee positivity.

Parameter accounting, worth memorising: 15,505,804 total, of which 14,405,124 — 92.9 percent —
are the two image encoders. Everything that actually fuses and predicts is about 1.1 million
parameters. That imbalance is central to the results.

The asterisk matters. The variant named CNN-only in the code is not image-only: the attention
query is still the BiLSTM output, so the temporal branch is still trained and still drives the
pooling. I found this while auditing and renamed it throughout the deck to "image + temporal
query". If an examiner catches a mislabelled ablation you have lost the room; better to
surface it yourself.
""")

# ═══════════════════════════════════════ 7 · MATH ════════════════════════════
s = S("What the gate, the loss and the metrics actually compute",
      "Formulation")
def eqbox(x, y, w, h, title_, eq, expl, color=NAVY):
    panel(s, x, y, w, h)
    txt(s, x+0.18, y+0.12, w-0.36, 0.24, title_, size=11, color=color, bold=True)
    txt(s, x+0.18, y+0.42, w-0.36, 0.4, eq, size=13.5, color=INK, italic=True, spacing=1.1)
    txt(s, x+0.18, y+h-0.72, w-0.36, 0.66, expl, size=9.5, color=GREY, spacing=1.03)
eqbox(0.55, 1.42, 3.95, 1.5, "1 · Clear-sky index — the physical normaliser",
      "k(t)  =  GHI(t) / GHI_clear(t)",
      "GHI_clear from pvlib's Ineichen model (solar geometry + turbidity climatology). "
      "k removes the deterministic diurnal and seasonal signal, leaving the cloud "
      "attenuation the model must actually predict.")
eqbox(4.68, 1.42, 3.95, 1.5, "2 · Physics gate",
      "α  =  σ( W₂ · ReLU(W₁ g + b₁) + b₂ )",
      "g = [k(t), cc/100, v_x, v_y] ∈ ℝ⁴ — clear-sky index, cloud fraction, and Farnebäck "
      "flow over the Singapore RoI. W₁ ∈ ℝ^{32×4}, W₂ ∈ ℝ^{1×32}: 193 parameters.")
eqbox(8.82, 1.42, 3.96, 1.5, "3 · Gated fusion",
      "h  =  α·H_t  +  (1 − α)·H_a",
      "H_t ∈ ℝ²⁵⁶ is the pooled BiLSTM state; H_a ∈ ℝ²⁵⁶ is the cross-attention output. "
      "α → 1 trusts the weather history; α → 0 trusts the imagery.")
eqbox(0.55, 3.08, 3.95, 1.62, "4 · Cross-attention (8 heads)",
      "H_a = softmax(QKᵀ/√d)·V,  Q = W_q H_t",
      "K = V = W_kv P with P ∈ ℝ^{196×352} the concatenated patch tokens from three global "
      "frames and one RoI frame; d = 256/8 = 32 per head. One query token, so attention "
      "returns a single pooled vector plus a 196-way weight map.")
eqbox(4.68, 3.08, 3.95, 1.62, "5 · Training objective — Gaussian NLL",
      "L = Σ_h [ log σ_h + (y_h − μ_h)² / (2σ_h²) ]",
      "Heteroscedastic likelihood, summed over the three horizons and averaged over the "
      "batch. The model is rewarded for widening σ exactly when it cannot predict y — this "
      "is what makes the intervals on slide 14 informative rather than decorative.")
eqbox(8.82, 3.08, 3.96, 1.62, "6 · Reference forecast and skill",
      "ŷ_SP(t+h) = k(t)·GHI_clear(t+h);   S = 1 − MAE/MAE_SP",
      "Smart persistence holds the clear-sky index constant and lets the geometry evolve. "
      "S > 0 means the model beats it; S = 0 means the model has learned nothing beyond "
      "the sun's position and the current cloud state.")
panel(s, 0.55, 4.88, 12.23, 1.42, fill=RGBColor(0xEC,0xF1,0xF7))
txt(s, 0.78, 5.02, 5.0, 0.24, "7 · PROBABILISTIC EVALUATION", size=10.5, color=NAVY, bold=True)
txt(s, 0.78, 5.32, 5.9, 0.9,
    "PICP  =  (1/N) Σ 1[ ŷ − 1.645σ ≤ y ≤ ŷ + 1.645σ ]        nominal 90 %\n"
    "PINAW =  mean interval width = 2 × 1.645 × σ̄   (W/m²)",
    size=12, color=INK, italic=True, spacing=1.25)
txt(s, 7.0, 5.32, 5.6, 0.95,
    "CRPS(y; μ, σ) = σ·[ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y−μ)/σ\n"
    "A proper score: rewards a sharp forecast only if it is also calibrated, so it cannot "
    "be gamed by widening σ the way PICP can.",
    size=12, color=INK, italic=True, spacing=1.2)
txt(s, 0.55, 6.42, 12.23, 0.6,
    "Denormalisation for reporting:  GHI = μ·σ_GHI + μ_GHI  with μ_GHI = 476.89, "
    "σ_GHI = 258.44 W/m² from train_stats.json; the 90 % band is rebuilt around the "
    "clamped mean so that the physics cap narrows the interval rather than collapsing it.",
    size=10.5, color=GREY, italic=True, spacing=1.06)
note(s, """
Only include an equation if you can defend every symbol — so be ready on all seven.

Equation 1, the clear-sky index. This is the single most important modelling choice in the
project. Raw GHI is dominated by a deterministic signal — where the sun is. Dividing by the
clear-sky value removes it and leaves the stochastic part, cloud attenuation. It is also
what makes the smart-persistence baseline strong, and it is the first input to the gate.

Equation 2 and 3, the gate. Four physical scalars in, one alpha out, and the fusion is a
convex combination. The design intent was that in clear conditions the model should lean on
the weather history, and in cloudy or fast-advecting conditions it should lean on the
imagery. Slide 15 shows the gate learned the right direction but with almost no dynamic
range.

Equation 4, cross-attention. Emphasise the shape: one query token, 196 key-value tokens.
This is not self-attention over a sequence of images; it is a learned pooling of image
patches conditioned on the temporal state. That is why the so-called CNN-only ablation still
depends on the BiLSTM.

Equation 5, the loss. If asked "why not MSE?", the answer is that MSE gives a point estimate
and this application needs an interval — a grid operator sizing reserve needs to know how
uncertain the forecast is. The Gaussian NLL learns mean and variance jointly. The known
failure mode is variance collapse early in training, which is why sigma is softplus plus
1e-4 and gradients are clipped at norm 1.

Equation 6, skill. Insist on this as the currency: MAE alone is meaningless across
different test periods, because an easier month gives a lower MAE for free. Skill against
smart persistence is normalised by exactly that difficulty.

Equation 7. PICP alone can be gamed — predict a huge sigma and coverage goes to 100 percent —
so it is always reported next to PINAW, the width. CRPS is the proper score that resolves
the tension, and it is reported for every variant on slide 14. Both CRPS and interval width
were absent from the original committed results; I added them during this analysis.
""")

# ═══════════════════════════════════════ 8 · TRAINING ════════════════════════
s = S("Two-phase schedule, probabilistic objective, early stopping",
      "Training")
pic(s, 0.55, 1.45, FIG+'f9_training.png', w=7.6)
caption(s, 0.55, 3.95, 7.6,
        "Figure 2 — full physics-gated model, 44 epochs on an RTX 4090 (16.7 min). "
        "Validation MAE is the t+1 h horizon on 914 samples.")
bullets(s, 0.55, 4.35, 7.7, 2.5, [
 (0, [("Reading the curves.  ", {'bold':True,'color':NAVY}),
      ("Both losses fall together to epoch 20 while the CNN is frozen. At epoch 20 the "
       "encoders are unfrozen at one tenth the learning rate: training loss accelerates "
       "downward, validation loss turns and rises. Everything after epoch 24 is "
       "memorisation of 6,675 samples by 15.5 M parameters.", {})]),
 (0, [("Why the checkpoint is taken at epoch 24.  ", {'bold':True,'color':NAVY}),
      ("Selection is on validation NLL, not MAE — the model is scored as a distribution, "
       "so the saved weights are the best calibrated ones, not merely the sharpest.", {})]),
 (0, [("Run-to-run variability.  ", {'bold':True,'color':RED}),
      ("This configuration was trained twice in the same session; best validation MAE was "
       "74.0 and 78.5 W/m². A 4.5 W/m² spread from stochasticity alone is larger than "
       "several of the gaps between architectures on slide 11.", {})]),
], size=12.5)
panel(s, 8.45, 1.45, 4.33, 5.4)
txt(s, 8.68, 1.62, 3.9, 0.26, "CONFIGURATION  (notebook cell 23)", size=10.5, color=ACC, bold=True)
cfg = [("Objective", "Gaussian NLL over 3 horizons"),
       ("Optimiser", "AdamW,  lr 3 × 10⁻⁵"),
       ("Weight decay", "1 × 10⁻¹  (deliberately high)"),
       ("Scheduler", "ReduceLROnPlateau, ×0.5, patience 8"),
       ("Batch size", "128"),
       ("Max epochs", "150"),
       ("Early stopping", "patience 20 (grid runs: 15)"),
       ("Gradient clipping", "global norm 1.0"),
       ("Mixed precision", "fp16 autocast + GradScaler"),
       ("CNN phase 1", "frozen, epochs 1–20 → 1,100,680 trainable"),
       ("CNN phase 2", "unfrozen at lr/10 → 15,505,804 trainable"),
       ("Encoder init", "SWIMSEG-pretrained B2 (else ImageNet)"),
       ("Seeds", "torch.manual_seed(42), np.random.seed(42)"),
       ("Hardware", "NVIDIA RTX 4090, 4 dataloader workers"),
       ("Checkpoint rule", "lowest validation NLL, saved with a config sidecar")]
yy = 1.95
for k, v in cfg:
    txt(s, 8.68, yy, 1.55, 0.3, k, size=9.5, color=GREY)
    txt(s, 10.3, yy, 2.35, 0.34, v, size=9.5, color=INK, bold=True, spacing=1.0)
    yy += 0.325
txt(s, 8.68, 6.9-0.08, 3.9, 0.3, "", size=9)
note(s, """
Explain the two-phase schedule as a deliberate decision, not a default. With 6,675 training
samples and 14.4 million CNN parameters, fine-tuning the encoders from step one destroys the
pretrained features before the small fusion head has learned anything useful. So the CNN is
frozen for twenty epochs — only 1.1 million parameters train — and then unfrozen at one tenth
the learning rate.

The curves show the schedule working and then the capacity problem asserting itself. Up to
epoch 20 train and validation move together. After unfreezing, the training loss keeps
falling and validation turns upward within four epochs. That divergence is the clearest
single piece of evidence for the overfitting story that runs through the whole results
section.

Two design points worth defending. Weight decay is 0.1, which is high; that is a response to
the same capacity problem. Checkpoint selection is on validation NLL rather than MAE, because
the deliverable is a calibrated distribution — selecting on MAE would favour a confident,
badly calibrated model.

The variability number is the one to volunteer rather than hide. The same configuration
trained twice gave best validation MAE of 74.0 and 78.5. Different points in the RNG stream
and slightly different early-stopping patience. That 4.5 W/m2 is my empirical noise floor,
and it is why on slide 13 I only claim differences that are both larger than this and
statistically significant.

Likely question: "why learning rate 3e-5, that is very low?" It is low because most of the
network is a pretrained CNN being fine-tuned; a higher rate destabilised the sigma head under
fp16 in early experiments. Related known weakness: batches producing non-finite loss are
skipped silently in the training loop and never counted, so I cannot report how often it
happened. That is on the limitations slide.
""")

# ═══════════════════════════════════════ 9 · EXPERIMENTAL DESIGN ═════════════
s = S("Six trained variants against five reference forecasters",
      "Experimental setup")
table(s, 0.55, 1.45, 7.7,
      ["Variant", "What it isolates", "Params", "Epochs", "GPU time"],
      [["Physics-Gated (large) ★", "the full proposed system", "15.51 M", "35", "12.7 min"],
       ["LSTM-only", "removes the image branch entirely", "15.51 M*", "150", "44.0 min"],
       ["Image + temporal query", "removes the gate, keeps attention", "15.51 M", "36", "13.1 min"],
       ["Naive concat fusion", "replaces the gate with concatenation", "15.57 M", "35", "12.6 min"],
       ["Physics-Gated (small)", "tests the over-capacity hypothesis", "7.49 M", "63", "21.9 min"],
       ["Physics-Gated (no ghi_lag1)", "removes the persistence shortcut", "15.51 M", "38", "14.4 min"]],
      col_w=[2.6,2.75,0.95,0.7,0.95], size=9.8, row_h=0.3,
      align=[PP_ALIGN.LEFT,PP_ALIGN.LEFT,PP_ALIGN.CENTER,PP_ALIGN.CENTER,PP_ALIGN.CENTER])
caption(s, 0.55, 3.66, 7.7,
        "* encoders exist but never reach the fusion path under ablation='lstm'. Epochs are from "
        "the grid runs (patience 15) — the curve on the previous slide is a separate patience-20 "
        "run of the same configuration. Grid cost 118.7 min on one RTX 4090.")
table(s, 0.55, 4.22, 7.7,
      ["Reference forecaster", "Role"],
      [["Persistence  (GHI held)", "floor — any model must beat it decisively"],
       ["Smart persistence  (k_t held)", "the operational reference; skill is measured against it"],
       ["Linear Regression", "does the problem need non-linearity?"],
       ["Random Forest  (100 trees)", "strong tabular learner, 14 features"],
       ["LightGBM  (200 estimators)", "strong tabular learner, 14 features"]],
      col_w=[2.9,4.8], size=9.8, row_h=0.29)
caption(s, 0.55, 5.98, 7.7,
        "Tabular baselines see 14 features at time t only — including ghi_lag1-3 — so they "
        "are given the same information the deep models get, minus the imagery.")
panel(s, 8.45, 1.45, 4.33, 5.4)
txt(s, 8.68, 1.62, 3.9, 0.26, "PROTOCOL", size=10.5, color=ACC, bold=True)
bullets(s, 8.68, 1.95, 3.92, 4.8, [
 (0, [("Split.  ", {'bold':True}), ("Chronological 70/15/15, never shuffled. Training "
      "ends 2025-11-02; nothing after that date is seen during fitting.", {})]),
 (0, [("Normalisation.  ", {'bold':True}), ("Statistics from the training split only, "
      "applied unchanged to validation and test.", {})]),
 (0, [("Leakage control.  ", {'bold':True}), ("pv_actual (r = 0.9999 with GHI) is excluded "
      "from all feature sets. Only future clear-sky — pure geometry — is given ahead of time.", {})]),
 (0, [("Identical rows.  ", {'bold':True}), ("Every number on slides 11–16 is computed on "
      "the same 914 samples, for deep models, tabular models and persistence alike.", {})]),
 (0, [("Metrics.  ", {'bold':True}), ("MAE, RMSE, R², skill vs smart persistence, PICP, "
      "interval width, CRPS — plus paired Diebold–Mariano tests.", {})]),
 (0, [("Seeds.  ", {'bold':True}), ("One run per variant. This is the design's main "
      "weakness and it is treated as such throughout.", {'color':RED})]),
], size=10.5, gap=9)
note(s, """
The variant table is built so that each row answers exactly one question, and I should be
able to say which question without looking.

LSTM-only versus the full model answers RQ1: does imagery help at all? Naive concat versus
the full model answers RQ2: is the gate better than just concatenating? Image-plus-temporal-
query isolates the attention path. The small preset tests whether the model is simply too
large for the data. The no-lag variant removes ghi_lag1, the previous hour's irradiance,
which is the strongest single predictor — it tells us how much of the performance is really
just persistence dressed up.

On the baselines: the tabular models get fourteen features including three lags of GHI. That
is deliberately generous. If I gave them less, beating them would prove nothing.

Be honest about cost when asked. The whole grid is under two hours of GPU time. That means
running three seeds per variant would have cost about six hours, which I did not spend, and
that is the single highest-value experiment left undone. I say so on the limitations slide
rather than waiting to be asked.

If challenged on epoch counts differing across variants — LSTM-only ran 150 epochs while the
others stopped between 35 and 63 — the answer is that early stopping is on validation NLL
with fixed patience, so the schedule is the same policy applied to different loss
trajectories. LSTM-only simply kept improving, which is itself informative: the smallest
effective model was the one that had not saturated.
""")

# ═══════════════════════════════════ 10 · EVALUATION INTEGRITY ═══════════════
s = S("A defect in my own results: the test split collapsed to 18 samples",
      "Evaluation integrity", rule=RED)
panel(s, 0.55, 1.4, 12.23, 0.86, fill=RGBColor(0xFB,0xEC,0xEC))
txt(s, 0.8, 1.52, 11.8, 0.6,
    "The committed results file results/fusion_ghi_comparison.csv reports every deep-model "
    "metric on 18 samples drawn from just two days, while results/baseline_ghi_comparison.csv "
    "reports the tabular models on 1,437. The two tables were never comparable — and the "
    "18-sample table cannot support any conclusion at all.",
    size=13, color=RED, bold=True, spacing=1.08)
bullets(s, 0.55, 2.42, 6.05, 4.3, [
 (0, [("Mechanism.  ", {'bold':True,'color':NAVY}),
      ("The 70/15/15 split is positional over all 9,574 rows. The image requirement is "
       "applied afterwards, inside the dataset class. Because the weather table was "
       "extended to August 2026 while the satellite archive stopped on 2026-06-07, the "
       "image-less tail fell almost entirely inside the test window.", {})]),
 (0, [("What the 18 samples are.  ", {'bold':True,'color':NAVY}),
      ("Two days only — 2026-05-15 and 2026-06-07 — with mean GHI 318.5 W/m² against "
       "492.8 for the full test split. Unusually dark, cloudy hours, evaluated in a "
       "single batch.", {})]),
 (0, [("Why it was not caught.  ", {'bold':True,'color':NAVY}),
      ("The notebook printed “Test: 18 | Test batches: 1” and the run continued. There "
       "was no assertion on split size.", {})]),
 (0, [("The fix, verified.  ", {'bold':True,'color':TEAL}),
      ("Filter to imaged rows before splitting, and assert a minimum test size. Recomputed "
       "on this dataset that yields 1,123 test samples over 116 days, 2025-10-13 → 2026-06-07 "
       "— which would also close the single-season limitation.", {})]),
 (0, [("What this deck does instead.  ", {'bold':True,'color':ACC}),
      ("Retraining was out of scope for the submission window, so every result from slide 11 "
       "onward is reported on the 914 imaged validation samples — 93 days — with the "
       "selection caveat restated each time.", {})]),
], size=11.4, gap=6)
pic(s, 6.85, 2.5, FIG+'f4_rank_flip.png', w=5.95)
caption(s, 6.85, 4.72, 5.95,
        "Figure 3 — the same six checkpoints, scored on 18 samples (left) and on 914 (right). "
        "The ordering inverts: the variant that looks best on 18 samples is fourth on 914, "
        "and the weather-only model moves from worst to best.")
panel(s, 6.85, 5.25, 5.93, 1.5, fill=RGBColor(0xEC,0xF1,0xF7))
txt(s, 7.08, 5.4, 5.5, 1.2,
    "Methodological lesson, and the one I would defend hardest:  a 5–15 W/m² ordering "
    "between architectures is not a result unless the sample size and the run-to-run "
    "variance can support it. At n = 18 the ordering is noise; at n = 914 it is stable "
    "and, on slide 13, statistically testable.",
    size=11.5, color=NAVY, spacing=1.1)
note(s, """
This is the slide that will decide how the examiners read everything that follows, so do not
soften it and do not rush it.

State the defect first, in one sentence: the deep-model results in the committed CSV were
computed on eighteen samples from two days, and they are not defensible. I found this by
re-reading my own notebook output during preparation — the warning was printed, "Test: 18,
Test batches: 1", and I had not acted on it.

Then the mechanism, because that is the interesting part. The split is positional: take the
first seventy percent of rows, then fifteen, then fifteen. The image requirement is applied
later, inside the dataset class, when it builds valid_indices. Those two facts interact badly
the moment image coverage becomes non-uniform in time. My weather table ran ahead to August
2026 while the satellite archive stalled in June, so the image-less rows piled up at the end
of the record — precisely the test split.

The fix is one line — drop image-less rows before splitting — plus an assertion so it can
never pass silently again. I verified what that fix yields on this dataset: about 1,123 test
samples spanning seven months. I did not retrain, because that is a full grid re-run, and I
say so plainly rather than presenting a fix I have not executed.

What I did instead is the honest compromise: report everything on the 914 imaged validation
samples. Be upfront about what that costs. The validation split was used for early stopping
and checkpoint selection, so these numbers are optimistic and are not a clean held-out test.
They are, however, sixty-six days across two months rather than two days, and — critically —
every model in the comparison is scored on identical rows.

The figure is the argument in one image. On eighteen samples the naive concat variant looks
best and the weather-only model looks worst. On nine hundred and fourteen the order reverses.
Same checkpoints, same weights. If an examiner takes one thing from this defence, this is it.
""")

# ═══════════════════════════════════════ 11 · MAIN RESULTS ═══════════════════
s = S("Main results: every model on the identical 914 samples",
      "Results")
table(s, 0.55, 1.45, 6.75,
      ["Model", "t+1 h", "t+2 h", "t+3 h", "mean", "skill"],
      [[("LightGBM", {'color':NAVY}), "63.4", "80.1", "89.6", "77.7", "+31.5 %"],
       [("Random Forest", {'color':NAVY}), "63.0", "81.6", "90.2", "78.3", "+31.0 %"],
       ["LSTM-only  (best deep)", "72.4", "82.7", "92.7", "82.6", "+27.2 %"],
       ["Naive concat fusion", "73.6", "83.9", "92.8", "83.4", "+26.5 %"],
       ["Physics-Gated (no ghi_lag1)", "76.5", "84.2", "92.4", "84.4", "+25.6 %"],
       ["Image + temporal query", "73.8", "86.2", "93.5", "84.5", "+25.5 %"],
       [("Physics-Gated (large) ★", {'color':ACC}), "78.5", "86.0", "95.0", "86.5", "+23.8 %"],
       ["Physics-Gated (small)", "101.1", "105.9", "109.5", "105.5", "+7.0 %"],
       ["Linear Regression", "83.7", "107.5", "123.7", "105.0", "+7.5 %"],
       [("Smart persistence", {'color':GREY}), "78.9", "117.5", "143.9", "113.5", "0"],
       [("Persistence", {'color':GREY}), "155.7", "253.9", "316.5", "242.0", "−113 %"]],
      col_w=[2.5,0.8,0.8,0.8,0.8,1.05], size=9.6, row_h=0.235, head_h=0.28,
      highlight=[0,2,6],
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*5)
caption(s, 0.55, 4.42, 6.75,
        "MAE in W/m² on the 914 imaged validation samples, 2025-11-03 → 2026-02-03. "
        "Skill = 1 − MAE / MAE(smart persistence) on the mean over the three horizons.")
pic(s, 7.75, 1.42, FIG+'f11_horizon.png', w=4.9)
caption(s, 7.55, 4.42, 5.25,
        "Figure 4 — MAE against horizon. The learned models separate from smart persistence "
        "immediately and the gap between trees and networks narrows with horizon.")
line(s, 0.55, 4.85, 12.78, 4.85, color=LGREY, lw=0.8)
bullets(s, 0.55, 4.98, 6.05, 2.0, [
 (0, [("OBSERVATION  ", {'bold':True,'color':TEAL,'size':9.5}),
      ("Every learned model beats smart persistence, and the margin widens with horizon.", {})]),
 (0, [("OBSERVATION  ", {'bold':True,'color':TEAL,'size':9.5}),
      ("Gradient-boosted trees are the most accurate models here — LightGBM 77.7 vs 86.5 W/m² "
       "for the full fusion network, with 0.03 % of its parameters.", {})]),
 (0, [("OBSERVATION  ", {'bold':True,'color':TEAL,'size':9.5}),
      ("Among the deep variants the ordering is monotone in how much image machinery is "
       "active: LSTM-only 82.6 < concat 83.4 < attention 84.5 < gated 86.5.", {})]),
], size=10.8, gap=6)
bullets(s, 6.95, 4.98, 5.85, 2.0, [
 (0, [("INTERPRETATION  ", {'bold':True,'color':NAVY,'size':9.5}),
      ("6,675 training samples cannot support 14.4 M image-encoder parameters. The imagery "
       "adds variance faster than it adds signal, and the more elaborate the fusion, the worse "
       "the result gets.", {})]),
 (0, [("INTERPRETATION  ", {'bold':True,'color':NAVY,'size':9.5}),
      ("The small preset is not a counter-example: shrinking the whole network also removed "
       "capacity the temporal branch needed, costing 23 W/m².", {})]),
 (0, [("SPECULATION  ", {'bold':True,'color':RED,'size':9.5}),
      ("A frozen encoder with a light probe, or far more training data, might reverse this. "
       "Nothing here tests that.", {})]),
], size=10.8, gap=6)
note(s, """
Lead with the positive result, because it is real and it is the one that generalises: every
learned model beats smart persistence, and the margin grows with horizon — from about twenty
percent at t+1h to nearly forty percent at t+3h for the best models. That is the physically
expected pattern. Persistence decays fast because cloud fields decorrelate; a learned model
holds on longer.

Then the finding that will get pushed back on: LightGBM at 77.7 W/m2 beats the full fusion
network at 86.5, using roughly five thousand decision trees against fifteen and a half
million parameters. Do not apologise for it and do not explain it away — quantify it. The gap
is 8.8 W/m2 on the mean, and slide 13 shows it is statistically significant at t+1h and not
significant at t+2h and t+3h. That nuance is the actual result.

The internal ordering of the deep variants is the sharpest evidence. Weather-only is best.
Adding concatenated image features costs about a watt. Adding cross-attention costs two.
Adding the physics gate on top costs four. Performance degrades monotonically with the amount
of image machinery in the fusion path. That is a coherent capacity story, not a random
ordering.

Keep the epistemic labels visible when you speak: what I observed, what I infer, and what is
speculation. If asked whether more data would fix it, say clearly that nothing in this project
tests that — it is a hypothesis, and I have marked it as one.

Have the parameter comparison ready as a soundbite: the gate itself is 193 parameters, the
whole fusion and prediction path is about 1.1 million, and the image encoders are 14.4
million — ninety-three percent of the model, for a branch that makes the forecast worse.
""")

# ═══════════════════════════════════ 12 · SKILL & HORIZON ════════════════════
s = S("Skill against smart persistence grows with horizon",
      "Results · skill analysis")
pic(s, 0.55, 1.45, FIG+'f3_skill.png', w=5.5)
caption(s, 0.55, 5.26, 5.5,
        "Figure 5 — skill score S = 1 − MAE/MAE_SP per horizon, 914 samples.")
table(s, 6.35, 1.5, 6.43,
      ["Model", "S(t+1 h)", "S(t+2 h)", "S(t+3 h)", "trend"],
      [["LightGBM", "+19.6 %", "+31.8 %", "+37.7 %", "rising"],
       ["Random Forest", "+20.2 %", "+30.6 %", "+37.3 %", "rising"],
       ["LSTM-only", "+8.2 %", "+29.6 %", "+35.6 %", "rising"],
       ["Naive concat fusion", "+6.7 %", "+28.6 %", "+35.5 %", "rising"],
       ["Physics-Gated (large)", "+0.5 %", "+26.8 %", "+34.0 %", "rising"],
       ["Physics-Gated (small)", "−28.1 %", "+9.9 %", "+23.9 %", "rising"],
       ["Linear Regression", "−6.1 %", "+8.5 %", "+14.0 %", "rising"]],
      col_w=[2.35,1.02,1.02,1.02,1.02], size=10, row_h=0.29, highlight=[0],
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*4)
bullets(s, 6.35, 4.05, 6.43, 2.9, [
 (0, [("OBSERVATION.  ", {'bold':True,'color':TEAL}),
      ("Skill rises monotonically with horizon for every model. At t+1 h the physics-gated "
       "network is statistically indistinguishable from smart persistence (+0.5 %); by "
       "t+3 h it removes a third of that baseline's error.", {})]),
 (0, [("INTERPRETATION.  ", {'bold':True,'color':NAVY}),
      ("This is the physically expected signature. At one hour ahead, holding the clear-sky "
       "index constant is already close to optimal — cloud fields have barely decorrelated, "
       "so there is little for a model to add. As the horizon lengthens the persistence "
       "assumption decays faster than the learned dynamics, and the learned models pull away.", {})]),
 (0, [("WHY IT MATTERS.  ", {'bold':True,'color':ACC}),
      ("Skill score is the standard currency in solar forecasting precisely because MAE is "
       "not comparable across test periods — an easier month lowers everyone's MAE. Skill "
       "normalises by the difficulty of the period, so these numbers are the ones that "
       "can be compared with published work.", {})]),
], size=11.8)
note(s, """
This is the slide to lead with if I only get to show one result.

The message: every learned model beats the operational reference, and the advantage grows
with horizon. LightGBM removes about twenty percent of smart persistence's error at one hour
and nearly thirty-eight percent at three. The best deep model tracks the same curve a few
points lower.

Explain why the curve rises, because that is the physics check. Smart persistence assumes the
cloud state is frozen and only the sun moves. Over one hour in Singapore that is nearly true —
the cloud field has barely decorrelated, so persistence is genuinely hard to beat, and the
physics-gated model only matches it, plus half a percent. Over three hours the frozen-cloud
assumption falls apart, while a model that has learned diurnal and seasonal structure and
autocorrelation degrades much more slowly. A model whose skill curve went the other way would
be a red flag that something is leaking.

Stress why I report skill rather than raw MAE as the headline. My own results show why: the
same random forest scores MAE 61.7 on the full test window and 78.3 on the validation window.
Same model, same training data, different period difficulty. Skill normalises that away, which
is why it is the standard in the solar forecasting literature and why every comparison here
carries it.

If asked which number to quote as the project's result, my answer is: about thirty percent
mean skill against smart persistence at one to three hours ahead on a single equatorial site,
achieved by a gradient-boosted tree, with the deep fusion model a few points behind.
""")

# ═══════════════════════════════════ 13 · ABLATION & SIGNIFICANCE ════════════
s = S("Ablations with a significance test — which differences are real?",
      "Ablation study")
txt(s, 0.55, 1.4, 12.23, 0.3,
    "Paired Diebold–Mariano tests on absolute-error loss, Harvey–Leybourne–Newbold "
    "small-sample correction, Newey–West lag h−1, n = 914.", size=11, color=GREY, italic=True)
table(s, 0.55, 1.78, 12.23,
      ["Comparison", "horizon", "MAE A", "MAE B", "DM statistic", "p-value", "verdict"],
      [[("LSTM-only (A)  vs  Physics-Gated (B)", {'color':NAVY}), "t+1 h", "72.4", "78.5",
        "−3.87", "0.0001", ("A better — significant", {'color':TEAL})],
       ["LSTM-only  vs  Physics-Gated", "t+2 h", "82.7", "86.0", "−2.34", "0.019",
        ("A better — significant", {'color':TEAL})],
       ["LSTM-only  vs  Physics-Gated", "t+3 h", "92.7", "95.0", "−1.87", "0.062",
        ("not significant", {'color':GREY})],
       [("LightGBM (B)  vs  LSTM-only (A)", {'color':NAVY}), "t+1 h", "72.4", "63.4",
        "+4.71", "2.9 × 10⁻⁶", ("B better — significant", {'color':ACC})],
       ["LightGBM  vs  LSTM-only", "t+2 h", "82.7", "80.1", "+1.17", "0.244",
        ("not significant", {'color':GREY})],
       ["LightGBM  vs  LSTM-only", "t+3 h", "92.7", "89.6", "+1.23", "0.219",
        ("not significant", {'color':GREY})],
       [("Physics-Gated (A)  vs  smart persistence (B)", {'color':NAVY}), "t+1 h", "78.5",
        "78.9", "−0.16", "0.872", ("no difference", {'color':RED})],
       ["Physics-Gated  vs  smart persistence", "t+2 h", "86.0", "117.5", "−8.44",
        "< 10⁻¹⁵", ("A better — significant", {'color':TEAL})],
       ["Physics-Gated  vs  smart persistence", "t+3 h", "95.0", "143.9", "−9.58",
        "< 10⁻¹⁵", ("A better — significant", {'color':TEAL})]],
      col_w=[3.9,0.85,0.8,0.8,1.15,1.05,2.1], size=9.7, row_h=0.255, head_h=0.29,
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*5+[PP_ALIGN.LEFT])
bullets(s, 0.55, 4.62, 6.1, 2.3, [
 (0, [("RQ1 — does imagery help?  ", {'bold':True,'color':NAVY}),
      ("No. Removing the image branch improves the forecast, and the improvement is "
       "significant at t+1 h (p = 0.0001) and t+2 h (p = 0.019). This is the project's "
       "central negative result, and it now rests on a paired test rather than on a "
       "single ordering.", {})]),
 (0, [("RQ2 — is the gate better than concatenation?  ", {'bold':True,'color':NAVY}),
      ("No. Naive concatenation (83.4) beats the gate (86.5) on the mean; the gate's "
       "193 parameters buy nothing measurable — see slide 15 for why.", {})]),
], size=11.8)
bullets(s, 6.9, 4.62, 5.88, 2.3, [
 (0, [("RQ3 — do trees beat the network?  ", {'bold':True,'color':NAVY}),
      ("At t+1 h yes, decisively. At t+2 h and t+3 h the difference is not statistically "
       "significant — the deep model closes the gap as the horizon lengthens, which is "
       "consistent with the sequence encoder contributing more when the lag features "
       "become less informative.", {})]),
 (0, [("Standing caveat.  ", {'bold':True,'color':RED}),
      ("Every DM test compares two single training runs. The test controls sampling noise "
       "in the evaluation set, not initialisation noise in training — measured at ≈4.5 W/m² "
       "on slide 8. Differences below that magnitude are not claimed.", {})]),
], size=11.8)
note(s, """
This slide converts orderings into claims, so be precise about what the test does and does
not do.

The Diebold-Mariano test is the standard tool for comparing two forecast series on the same
observations. It works on the loss differential — here the difference in absolute errors,
sample by sample — and asks whether its mean is distinguishable from zero, accounting for
autocorrelation, which matters because multi-step forecast errors are serially correlated. I
use the Newey-West variance with lag h minus one and the Harvey-Leybourne-Newbold small-sample
correction. Neither the original notebook nor the committed results contained any significance
test; I added this during preparation.

Read the three blocks. First: the weather-only model is significantly better than the full
physics-gated model at one and two hours. So the answer to research question one is a
statistically supported no. Second: LightGBM beats the deep model significantly at one hour
but not at two or three — the honest statement is "trees win clearly at short horizon, and
the models are indistinguishable further out." Third: the physics-gated model does not beat
smart persistence at one hour at all, p equals 0.87, but crushes it at two and three hours.

Volunteer the limitation before it is asked. The DM test controls for randomness in which
samples I evaluated on. It does not control for randomness in training. With one seed per
variant I cannot separate an architecture effect from an initialisation effect for
differences of a few watts. That is why I state the direction of the imagery result — which is
consistent across every comparison and every horizon — rather than defending its exact
magnitude.

If asked what would settle it: three to five seeds per variant, mean and standard deviation
reported, and the DM test run on seed-averaged errors. About six GPU-hours. It is the first
item on my future work list.
""")

# ═══════════════════════════════════ 14 · CALIBRATION ════════════════════════
s = S("Probabilistic calibration: the clearest success in the project",
      "Results · uncertainty")
pic(s, 0.55, 1.42, FIG+'f6_forecast.png', w=7.55)
caption(s, 0.55, 4.36, 7.55,
        "Figure 6 — LSTM-only, t+1 h, 120 consecutive validation samples with the 90 % "
        "predictive interval. The band widens on broken-cloud afternoons and narrows on "
        "clear mornings.")
table(s, 0.55, 4.86, 7.55,
      ["Variant", "PICP %", "width W/m²", "CRPS W/m²", "MAE W/m²"],
      [["LSTM-only", "90.8", "372", "59.8", "82.6"],
       ["Naive concat fusion", "90.5", "365", "60.1", "83.4"],
       ["Image + temporal query", "90.9", "366", "60.8", "84.5"],
       ["Physics-Gated (no ghi_lag1)", "90.0", "360", "61.1", "84.4"],
       ["Physics-Gated (large)", "91.9", "385", "62.0", "86.5"],
       ["Physics-Gated (small)", "91.4", "486", "76.3", "105.5"]],
      col_w=[2.75,1.0,1.3,1.25,1.25], size=9.6, row_h=0.25, head_h=0.28,
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*4)
caption(s, 0.55, 6.72, 7.55,
        "Nominal coverage is 90 %. CRPS ranks the variants identically to MAE — sharpness "
        "follows accuracy.")
pic(s, 8.35, 1.42, FIG+'f5_calibration.png', w=4.43)
caption(s, 8.35, 2.94, 4.43,
        "Figure 7 — coverage at n = 914 vs n = 18, and mean interval width.")
bullets(s, 8.35, 3.35, 4.43, 3.5, [
 (0, [("OBSERVATION.  ", {'bold':True,'color':TEAL}),
      ("Empirical coverage is 90.0–91.9 % against a nominal 90 % — every variant, with no "
       "post-hoc recalibration.", {})]),
 (0, [("INTERPRETATION.  ", {'bold':True,'color':NAVY}),
      ("The Gaussian NLL objective did its job: σ is learned per sample, so the model "
       "expresses genuine conditional uncertainty rather than a fixed band.", {})]),
 (0, [("Honest caveat.  ", {'bold':True,'color':RED}),
      ("Coverage without width is meaningless. 372 W/m² is 78 % of mean GHI — calibrated, "
       "but blunt at three hours ahead.", {})]),
 (0, [("Sample size again.  ", {'bold':True,'color':ACC}),
      ("The committed 18-sample table reports PICP 55–76 %, which reads as a calibration "
       "failure. At n = 914 the same checkpoints give 90–92 %. The failure was the test set.", {})]),
], size=10.8, gap=7)
note(s, """
Open by saying this is where the design choice paid off. The model was trained to emit a mean
and a standard deviation per horizon under a Gaussian negative-log-likelihood, and the result
is that ninety percent intervals contain the truth ninety to ninety-two percent of the time
across every variant, with no post-hoc recalibration, no conformal wrapper, nothing.

Point at the time series and describe the behaviour, not just the number. The band is narrow
on clear mornings and widens on broken-cloud afternoons. That is conditional uncertainty —
exactly what a heteroscedastic model is supposed to produce and what a fixed-width interval
cannot.

Then immediately undercut it, because an examiner will otherwise do it for you. Coverage on
its own is trivially gameable: predict an enormous sigma and coverage goes to one hundred
percent. So width is reported next to it, and the width is 372 W/m2, which is seventy-eight
percent of mean irradiance. Calibrated but blunt. For an operator sizing reserve, a band
that wide is honest but not very actionable at three hours.

CRPS is the resolution of that tension — it is a proper scoring rule, so it penalises both
miscalibration and unnecessary width. It ranks the variants in the same order as MAE, which
tells us none of the variants is buying accuracy by being over-confident.

Finally, connect back to slide 10. The committed results show coverage of fifty-five to
seventy-six percent, which looks like a calibration failure, and I could easily have written
a slide explaining that failure. It was not real. On 914 samples the same checkpoints are
calibrated. An eighteen-sample coverage estimate has a standard error of about seven
percentage points — it could not have detected anything.
""")

# ═══════════════════════════════════ 15 · GATE BEHAVIOUR ════════════════════
s = S("Why the gate did not help: right direction, 13 % of the range",
      "Mechanism analysis")
pic(s, 0.55, 1.45, FIG+'f7_gate.png', w=7.6)
caption(s, 0.55, 3.98, 7.6,
        "Figure 8 — gate output α for all 914 samples. Left: α against its own first input, "
        "the clear-sky index. Right: mean α by hour. Both panels are drawn on the full 0–1 "
        "range the sigmoid can reach.")
table(s, 0.55, 4.42, 7.6,
      ["Quantity", "Value", "Reading"],
      [["mean α", "0.489", "an almost exactly equal blend of the two branches"],
       ["standard deviation of α", "0.027", "the gate moves ±2.7 % over every condition seen"],
       ["observed range", "0.432 – 0.561", "13 % of the available (0, 1) interval"],
       ["corr(α, clear-sky index)", "+0.895", "clear sky → trust the weather history"],
       ["corr(α, cloud cover)", "−0.611", "cloudy → shift weight toward the imagery"],
       ["corr(α, hour of day)", "+0.435", "morning leans on imagery, afternoon on history"]],
      col_w=[2.4,1.35,3.85], size=9.8, row_h=0.27, highlight=[2])
bullets(s, 8.4, 1.45, 4.38, 5.4, [
 (0, [("OBSERVATION.  ", {'bold':True,'color':TEAL}),
      ("α correlates +0.90 with the clear-sky index and −0.61 with cloud cover — the "
       "physically intended direction, learned without supervision.", {})]),
 (0, [("OBSERVATION.  ", {'bold':True,'color':TEAL}),
      ("But α never leaves 0.43–0.56. In every condition the fusion is within 7 % of a "
       "fixed 50/50 average of the two branches.", {})]),
 (0, [("INTERPRETATION.  ", {'bold':True,'color':NAVY}),
      ("A gate that never commits cannot route. It behaves as a constant, so the "
       "architecture reduces in practice to a fixed mean of the temporal and attended-image "
       "vectors — which is why it performs slightly worse than explicit concatenation, "
       "where the head can at least learn its own weighting over 512 dimensions.", {})]),
 (0, [("INTERPRETATION.  ", {'bold':True,'color':NAVY}),
      ("Nothing in the objective rewards a decisive gate. The NLL is minimised by whatever "
       "blend reduces error on average, and if the image branch carries little signal, "
       "averaging it in at a constant weight is the safe optimum.", {})]),
 (0, [("SPECULATION.  ", {'bold':True,'color':RED}),
      ("An entropy or sparsity penalty on α, or a temperature on the sigmoid, might force "
       "commitment. Untested here.", {})]),
 (0, [("Caveat.  ", {'bold':True,'color':GREY}),
      ("Two of the gate's four inputs behave differently at serving time — see slide 17.", {})]),
], size=11.6)
note(s, """
This is the mechanistic explanation for the negative result on slides 11 and 13, and it is
where I show I understand my own model rather than just reporting its scores.

First, the good news, and it is genuinely good: the gate learned the physically correct
policy without ever being told to. Alpha correlates plus 0.90 with the clear-sky index and
minus 0.61 with cloud cover. In clear conditions it shifts weight toward the weather history;
in cloudy conditions toward the imagery. That is exactly the design intent, and it emerged
purely from the negative-log-likelihood objective.

Now the problem, which the left panel makes visually undeniable — and note I deliberately
plotted the full zero-to-one range rather than autoscaling, because autoscaling would have
made a flat line look like a strong trend. Alpha never leaves 0.43 to 0.56. Its standard
deviation is 0.027. Under every condition in sixty-six days of data, the fusion is within
seven percent of a fixed fifty-fifty average.

So the interpretation: the gate is directionally correct and operationally inert. It is a
constant with a very slight tilt. That explains why the gated variant is slightly worse than
naive concatenation — concatenation hands the head all 512 dimensions and lets it learn its
own weighting, while the gate collapses them into a near-fixed average first, destroying
information.

Why did it collapse? Nothing in the loss rewards decisiveness. If the image branch carries
little signal, a constant blend is the risk-minimising solution, and gradient descent finds
it. A sharper gate would only pay off if the image branch were sometimes much better than the
temporal branch — and slide 16 shows it only is, marginally, in the most overcast conditions.

If asked how to fix it: an entropy penalty on alpha, or a temperature parameter on the
sigmoid, or supervising alpha directly against a cloud-regime label. I mark that as
speculation because I did not run it.
""")

# ═══════════════════════════════════ 16 · ERROR ANALYSIS ═════════════════════
s = S("Where the forecasts fail — and where imagery finally helps",
      "Error analysis")
pic(s, 0.55, 1.45, FIG+'f8_errors.png', w=7.7)
caption(s, 0.55, 4.0, 7.7,
        "Figure 9 — MAE at t+1 h stratified by the clear-sky index at issue time (left) and "
        "by hour of day (right), 914 samples.")
table(s, 0.55, 4.34, 7.7,
      ["Regime  (k_t at issue time)", "n", "LSTM-only", "Physics-Gated", "LightGBM", "Smart pers."],
      [["< 0.3   overcast", "57", "89.0", ("80.9", {'color':TEAL}), "54.1", "83.7"],
       ["0.3–0.5  heavy cloud", "149", "81.6", "92.6", "67.1", "96.5"],
       ["0.5–0.7  broken cloud", "199", "82.9", "86.8", "75.1", "86.5"],
       ["0.7–0.9  light cloud", "300", "71.8", "74.9", "66.7", "73.6"],
       ["> 0.9   near clear", "209", "52.4", "65.0", "47.4", "65.5"]],
      col_w=[2.5,0.6,1.2,1.35,1.05,1.1], size=9.8, row_h=0.275,
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*5)
caption(s, 0.55, 6.06, 7.7,
        "MAE at t+1 h in W/m². Teal marks the only regime where the image-using model beats "
        "the weather-only model.")
bullets(s, 8.5, 1.45, 4.28, 5.4, [
 (0, [("Failure mode 1 — partly cloudy.  ", {'bold':True,'color':NAVY}),
      ("Error peaks in the 0.3–0.7 band, where a single convective cell decides whether the "
       "next hour is 300 or 800 W/m². The clear and overcast regimes are both easier "
       "because the outcome is more determined.", {})]),
 (0, [("Failure mode 2 — afternoon convection.  ", {'bold':True,'color':NAVY}),
      ("MAE at t+1 h rises from 54 W/m² at 09:00 to 107 W/m² at 14:00 — the diurnal "
       "convective build-up over the Malay peninsula, the hardest hours in the day.", {})]),
 (0, [("The one win for imagery.  ", {'bold':True,'color':TEAL}),
      ("In the overcast bin (k_t < 0.3) the physics-gated model beats the weather-only "
       "model, 80.9 vs 89.0 W/m². It is the only regime where it does, and n = 57 — "
       "suggestive, not established.", {})]),
 (0, [("Structural error source.  ", {'bold':True,'color':RED}),
      ("Targets are built with a positional shift, and the record keeps only 08:00–17:00. "
       "So 10 % / 20 % / 30 % of the t+1/2/3 h targets are the next morning, not the next "
       "hour. Restricted to genuine same-day pairs, t+1 h MAE rises 72.4 → 77.4 (LSTM-only) "
       "and 63.4 → 69.0 (LightGBM): every headline number is ~8 % optimistic. The ranking "
       "is unaffected.", {})]),
], size=11.3)
note(s, """
Three things to convey: where the error lives, why it lives there, and one structural defect
I found in my own target construction.

Where. Stratifying by the clear-sky index at issue time shows a clear inverted-U. Near-clear
conditions are easy, about 52 W/m2 for the weather-only model. Overcast is also relatively
easy — if it is thoroughly dark it will probably stay dark. The hard band is in between, 0.3
to 0.7, broken cloud, where a single convective cell decides whether the next hour is 300 or
800 W/m2. Physically that is exactly right: the error is highest where the outcome variance
is highest.

The hour-of-day panel says the same thing in the time domain. Error more than doubles from
mid-morning to two in the afternoon, tracking Singapore's diurnal convective build-up. If a
future version were to be improved anywhere, it would be here.

Now the one place the imagery earns its keep. In the most overcast bin the physics-gated model
beats the weather-only model by eight W/m2. That is the only regime where it wins, and it is
consistent with the mechanism: when the current cloud state is unambiguous and thick, an
upstream view carries information a point time series does not. I state it carefully — n is
fifty-seven, so it is suggestive and not established. It is also the most useful lead for
future work: gate on regime, and only invoke the image branch when it is likely to help.

Finally, the structural defect, which I found by checking timestamp differences rather than
row positions. Targets are built with a positional shift by one, two and three rows, but the
dataset only contains 08:00 to 17:00. So for the last rows of each day the shift crosses the
overnight gap: ten percent of t+1h targets, twenty percent at t+2h, thirty percent at t+3h are
actually the next morning. Those are easier to predict than a true one-hour-ahead value,
because the model can lean on the diurnal pattern. Restricting to genuine same-day pairs
raises every MAE by roughly eight percent. Crucially the ranking is unchanged — it affects
every model equally — but the absolute numbers are optimistic and I say so. The same
positional shift also corrupts the lag features and the frame offsets, which is a further
reason the image branch underperformed.
""")

# ═══════════════════════════════════ 17 · DEPLOYMENT ═════════════════════════
s = S("What runs daily — and the train/serve divergences in it",
      "Operational system", rule=TEAL)
panel(s, 0.55, 1.42, 6.05, 2.62)
txt(s, 0.78, 1.56, 5.6, 0.26, "THE LIVE PATH  (cron: predict 10:00 SGT, verify 13:30 SGT)",
    size=10, color=TEAL, bold=True)
steps = [
 ("current_data.py", "fetches frames t, t−1 h, t−2 h. Each tile is validated for brightness "
                     "and structure; a failed tile is retried at 10-min steps across up to "
                     "four sources."),
 ("model.py helpers", "load the CSV, top it up from the live API (the archive lags ~5 days), "
                      "build the 24-step window, normalise with train_stats.json."),
 ("load_model()", "reads the .config.json sidecar, so width, head count and ablation are "
                  "reconstructed exactly — head count is not recoverable from shapes."),
 ("apply_physics_cap()", "clamps μ to 1.15 × the preceding-hour-mean clear-sky GHI and "
                         "rebuilds the 90 % band around the clamped mean."),
 ("forecast_latest.json", "μ, 90 % interval, clear-sky value and diagnostics: image_ok, "
                          "outside_training_hours, lookback_ends, capped."),
]
yy = 1.9
for k, v in steps:
    txt(s, 0.78, yy, 1.45, 0.3, k, size=9.0, color=NAVY, bold=True, font="Consolas", spacing=1.0)
    txt(s, 2.3, yy, 4.15, 0.44, v, size=9.0, color=INK, spacing=0.98)
    yy += 0.42
panel(s, 6.8, 1.42, 5.98, 2.62, fill=RGBColor(0xF2,0xF8,0xF6))
txt(s, 7.03, 1.54, 5.5, 0.26,
    "LIVE EVIDENCE  ·  BEFORE AND AFTER THE SATELLITE FIXES OF 20 AUG 2026",
    size=9.5, color=TEAL, bold=True)
table(s, 7.03, 1.84, 5.55,
      ["", "MAE", "fc sd", "obs sd", "corr", "21 Aug", "actual", "error"],
      [["t+1 h", "65.0", "26.3", "61.5", ("−0.66", {'color':RED}), "148.4", "225.0", "76.6"],
       ["t+2 h", "88.3", "22.2", "63.2", ("−0.74", {'color':RED}), "351.9", "449.0", "97.1"],
       ["t+3 h", "85.0", "28.3", "87.6", ("−0.49", {'color':RED}), "512.6", "636.0", "123.4"]],
      col_w=[0.62,0.62,0.62,0.66,0.66,0.78,0.72,0.7], size=8.6, hsize=8.4,
      row_h=0.235, head_h=0.26,
      align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*7)
txt(s, 7.03, 2.86, 5.55, 1.1,
    "Left block — 8 daily 10:00 SGT forecasts, 12–19 Aug 2026, verified over 24 hours, before "
    "the tile-validation and three-frame fixes. MAE 79.4 W/m² looks respectable, but the "
    "forecast varied only ~36 % as much as the outcome and correlated negatively with it at "
    "every horizon: the model was nearly constant and the MAE was flattered by a calm week. "
    "Coverage was 100 % at a band ~540 W/m² wide.\n"
    "Right block — the first post-fix issue time (08:00 SGT, 21 Aug). All three inside the "
    "90 % band, all three low. Three verified hours cannot support a conclusion.",
    size=8.6, color=INK, spacing=1.0)
txt(s, 0.55, 4.2, 12.23, 0.28,
    "FOUR TRAIN/SERVE DIVERGENCES IDENTIFIED IN THE LIVE PATH", size=11, color=RED, bold=True)
table(s, 0.55, 4.55, 12.23,
      ["#", "Divergence", "Training behaviour", "Serving behaviour", "Consequence", "Status"],
      [["1", "Gate input 2 — cloud cover", "real ERA5 cloud fraction",
        "synthesised as (1 − k_t)×100", "gate input 2 becomes a deterministic function of "
        "input 1", ("open", {'color':RED})],
       ["2", "Lookback time base", "24 consecutive daylight rows ≈ 2.4 days",
        "24 consecutive clock hours", "at a 10:00 issue time ~14 of 24 rows are night hours "
        "the model never saw", ("open", {'color':RED})],
       ["3", "Station weather", "not used", "fetched from data.gov.sg, printed, written to "
        "JSON — never enters the tensor", "misleading pipeline documentation, no numerical "
        "effect", ("open", {'color':ACC})],
       ["4", "Verification source", "target = Open-Meteo shortwave radiation",
        "verify.py labels the same product “measured”", "verification is model-vs-analysis, "
        "not against an independent pyranometer", ("open", {'color':ACC})]],
      col_w=[0.35,2.05,2.3,2.85,3.4,0.75], size=9.1, row_h=0.42, head_h=0.28,
      align=[PP_ALIGN.CENTER]+[PP_ALIGN.LEFT]*5)
txt(s, 0.55, 6.66, 12.23, 0.3,
    "Divergences 1 and 2 are one-line fixes in model.py — identified and quantified but not "
    "yet corrected. With three verified post-fix hours, this deck makes no claim about live "
    "accuracy; a fortnight of corrected forecasts is the first future-work item.",
    size=10, color=NAVY, italic=True)
note(s, """
Two halves: what the deployment does well, and what is still wrong with it. Do not let the
first half sound like a claim of live accuracy.

What it does well. The satellite fetch is defensive by design: every tile is validated for
mean brightness and standard deviation, with the thresholds derived from the training images
themselves, and a failed tile triggers a walk back in ten-minute steps across up to four
sources. That guard exists because an earlier version of this system was silently forecasting
from a black "No Image" placeholder that the provider served with HTTP 200 — the status code
was fine, the pixels were not. Fourteen point four million parameters were being fed a
constant black square. That was the single worst defect in the project and it is fixed.

The config sidecar matters too: head count cannot be recovered from tensor shapes, so loading
a checkpoint with the wrong head count would silently produce garbage rather than raising an
error. Writing the sidecar makes the failure impossible.

The physics cap deserves a word because it contains a subtlety I got wrong first time. It
clamps the mean to 1.15 times the preceding-hour-mean clear-sky value — hour-mean, not
instantaneous, because Open-Meteo labels each hourly value with the end of its averaging
window. And it rebuilds the interval around the clamped mean rather than clipping mean, lower
and upper independently, which would collapse the interval to a point whenever the cap binds.

Now the divergences, and this is the part to present as a strength. Number one: at training
time the gate's second input is the real ERA5 cloud fraction; at serving time it is computed
as one minus the clear-sky index. So in deployment two of the four gate inputs are
deterministically related, which is not the function the gate was trained on. Number two is
probably worse: training used twenty-four consecutive daylight rows, spanning about two and a
half days; serving takes twenty-four clock hours, so at a ten in the morning issue time
roughly fourteen of the twenty-four rows are night, with irradiance near zero — a regime that
appears nowhere in training.

Both are one-line fixes and both are still open.

Now the live evidence table, which is the most interesting thing on the slide. The left block
is eight daily forecasts from the twelfth to the nineteenth of August, verified over
twenty-four hours, all issued before the tile-validation and three-frame fixes landed on the
twentieth. The mean absolute error is 79.4 W/m2, which sounds respectable — and it is
completely misleading. Look at the two standard-deviation columns: the forecast varied by
about twenty-five W/m2 while the outcome varied by about seventy. The model was emitting a
nearly constant number. And the correlation between forecast and outcome is negative at every
horizon, minus 0.66, minus 0.74, minus 0.49. A forecast that moves opposite to the outcome has
no skill at all; the low MAE came from a calm week, not from accuracy. Coverage was one
hundred percent, but with a band about 540 W/m2 wide — roughly the mean signal itself.

That is what a broken deployment looks like from the outside, and it is why I insist on
reporting interval width next to coverage and correlation next to MAE.

The right block is the first post-fix issue time, this morning at eight. All three horizons
landed inside the ninety percent band, and all three were low by seventy-seven to a hundred
and twenty-three W/m2. Three verified hours. I will not draw a conclusion from that, and I say
so on the slide. What I will say is that the diagnosis is complete and the instrumentation is
now in place to measure the fix — a fortnight of corrected forecasts is the first item of
future work.
""")

# ═══════════════════════════════════ 18 · COMPUTE & REPRODUCIBILITY ══════════
s = S("Computational cost, engineering trade-offs and reproducibility",
      "Systems & reproducibility")
txt(s, 0.55, 1.4, 6.0, 0.26, "COST AND CAPACITY", size=10.5, color=ACC, bold=True)
table(s, 0.55, 1.72, 6.05,
      ["Quantity", "Value", "Source"],
      [["Total parameters", "15,505,804", "measured"],
       ["  · image encoders (2 × B2)", "14,405,124  (92.9 %)", "measured"],
       ["  · BiLSTM + attention pooling", "572,673", "measured"],
       ["  · cross-attention", "419,328", "measured"],
       ["  · physics gate", "193", "measured"],
       ["  · enrich + head", "108,486", "measured"],
       ["Trainable, epochs 1–20", "1,100,680  (7.1 %)", "notebook cell 23"],
       ["Small preset", "7,489,216", "measured"],
       ["Checkpoint on disk", "62.9 MB", "best_model.pt"],
       ["Training, full model", "16.7 min / 44 epochs", "RTX 4090 log"],
       ["Full experiment grid", "118.7 min", "notebook cell 25"],
       ["Image cache", "7,654 × 224² tensors ≈ 4.6 GB RAM", "notebook cell 16"],
       ["Inference, batch 1, CPU", "755 ms  (fp32, Apple Silicon)", "measured for this deck"],
       ["LightGBM equivalent", "600 trees, 18,600 leaves, 1.5 MB, ≈1 ms", "measured for this deck"]],
      col_w=[2.75,2.15,1.15], size=9.4, row_h=0.245, head_h=0.28, highlight=[1,12])
bullets(s, 0.55, 5.65, 6.05, 1.3, [
 (0, [("Trade-off.  ", {'bold':True,'color':NAVY}),
      ("Against LightGBM the network costs ~700× the inference time and ~40× the model size, "
       "and is less accurate on this data. Neither latency is a constraint for an hourly "
       "forecast — the decisive argument is accuracy per parameter.", {})]),
], size=11.3)
txt(s, 6.95, 1.4, 5.8, 0.26, "REPRODUCIBILITY", size=10.5, color=ACC, bold=True)
panel(s, 6.95, 1.72, 5.83, 2.5)
cmds = [
 ("environment", "Python 3.11 (training host) · torch · timm · pvlib · lightgbm · opencv"),
 ("1  build the table", "python combined_dataset.py --tolerance 60"),
 ("2  train + evaluate", "run solar_pv_main.ipynb (cells 1–28) on a CUDA host"),
 ("3  live forecast", "python predict.py            → forecast_latest.json"),
 ("4  verify", "python verify.py             → verification_log.csv"),
 ("5  automate", "daily_run.sh predict | verify  (cron 10:00 / 13:30 SGT)"),
]
yy = 1.9
for k, v in cmds:
    txt(s, 7.15, yy, 1.55, 0.3, k, size=9.3, color=NAVY, bold=True)
    txt(s, 8.75, yy, 3.9, 0.36, v, size=9.2, color=INK, font="Consolas", spacing=1.0)
    yy += 0.4
bullets(s, 6.95, 4.35, 5.83, 2.6, [
 (0, [("Reproduced, not asserted.  ", {'bold':True,'color':TEAL}),
      ("For this deck the entire pipeline was re-derived independently from the CSV and the "
       "committed checkpoints. The split sizes (6,675 / 914 / 18), the normalisation "
       "statistics and all six reported test MAEs were reproduced to within 1 % — the "
       "residual is fp16 autocast in training versus fp32 here. Details in Appendix A4.", {})]),
 (0, [("Determinism gaps that remain.  ", {'bold':True,'color':RED}),
      ("Seeds are set but cuDNN benchmark mode and TF32 are enabled, and dataloader worker "
       "order is not seeded — so runs are not bit-identical. Empirically this is worth "
       "≈4.5 W/m² of validation MAE.", {})]),
 (0, [("Data versioning gap.  ", {'bold':True,'color':RED}),
      ("data/ is gitignored, so the exact CSV the checkpoints were trained on is not tracked "
       "in git; it must be rebuilt or copied. A frozen, hashed dataset file is the correct fix.", {})]),
], size=11.0, gap=8)
note(s, """
The parameter table is the argument, so walk down it rather than reading it out. Fifteen and
a half million parameters, of which ninety-three percent are the two image encoders. The
physics gate — the conceptual centrepiece of the architecture — is one hundred and ninety-three
parameters. The entire fusion and prediction path is about 1.1 million. So the model is,
by mass, almost entirely an image feature extractor for a branch that slide 13 shows makes
the forecast worse.

On timing: I measured 755 milliseconds per forecast on CPU at batch one, in float32, on this
machine — that is a measurement I took for this deck, not a repository number, and I label it
that way. LightGBM does the same job in under five milliseconds. For an hourly forecast
neither latency is a constraint; nobody cares about 750 milliseconds once an hour. The
meaningful trade-off is not latency, it is accuracy per parameter, and there the trees win
outright.

On reproducibility, make the strong claim precisely. I did not simply assert that the results
reproduce — I re-derived the entire pipeline independently from the CSV and the committed
checkpoints, and recovered the split sizes exactly, the normalisation statistics exactly, and
all six test MAEs to within one percent. The residual difference is that training and the
original evaluation ran under fp16 autocast while my reproduction ran fp32. That is in
appendix A4 with the numbers side by side.

Then be equally precise about what does not reproduce. Seeds are set, but cuDNN benchmark mode
and TF32 are on and dataloader workers are not seeded, so runs are not bit-identical — worth
about 4.5 W/m2. And the data directory is gitignored, so the exact CSV behind the checkpoints
is not tracked. Both are stated as defects with named fixes rather than glossed over.
""")

# ═══════════════════════════════════ 19 · LIMITATIONS ════════════════════════
s = S("Limitations — stated at the strength the evidence actually supports",
      "Limitations", rule=RED)
lims = [
 ("Evaluation", "The reported held-out test split collapsed to 18 imaged samples from two "
  "days, so all results here use the 914 imaged validation rows.",
  "Those rows drove early stopping and checkpoint selection, so every deep-model number is "
  "optimistic. This is not a clean held-out test.", RED),
 ("Statistics", "One training run per variant; no seed replication.",
  "Two runs of the same configuration differed by 4.5 W/m². Differences below that are not "
  "attributable to architecture. Three seeds × six variants ≈ 6 GPU-hours would settle it.", RED),
 ("Temporal scope", "The evaluation window is 2025-11-03 → 2026-02-03 — the north-east "
  "monsoon only.",
  "Nothing here shows the ranking holds through the south-west monsoon or the inter-monsoon "
  "periods. Blocked rolling-origin CV is the correct design.", RED),
 ("Target construction", "Targets, lag features and frame offsets use a positional shift on "
  "a table containing only 08:00–17:00.",
  "10 / 20 / 30 % of t+1/2/3 h targets cross the overnight gap, making every MAE ≈8 % "
  "optimistic; the same defect corrupts the previous-frame and optical-flow pairs.", ACC),
 ("Physics", "The clear-sky denominator is the instantaneous pvlib value while GHI is an "
  "hourly mean.",
  "Empirical max k_t drifts monotonically from 0.69 at 08:00 to 1.43 at 17:00 — a timestamp-"
  "convention mismatch. The hour-mean fix exists in the code but is applied only to the "
  "deployment cap, not to the dataset.", ACC),
 ("Ground truth", "The target is Open-Meteo/ERA5 shortwave radiation, and verify.py scores "
  "against the same product.",
  "No independent pyranometer is involved anywhere. The system is validated against a "
  "reanalysis, not against measured irradiance.", ACC),
 ("Generality", "One site, one satellite, ~6.7 k training hours, 2.5 years.",
  "The negative result about imagery is a statement about this data regime, not about "
  "satellite fusion in general.", NAVY),
 ("Deployment", "Two train/serve divergences remain open; live verification covers 24 "
  "pre-fix hours and 3 post-fix hours.",
  "The pre-fix log shows the forecast correlating negatively with the outcome at every "
  "horizon, so its 79.4 W/m² MAE is not evidence of skill. No claim is made about live "
  "accuracy.", NAVY),
 ("Instrumentation", "Non-finite training batches are skipped without being counted, and "
  "future_clearsky is fed unnormalised alongside z-scored features.",
  "The first hides a possible fp16 stability problem; the second lets three raw 0–1000 W/m² "
  "inputs dominate the scale of one layer.", NAVY),
]
yy = 1.42
for area, what, why, col in lims:
    txt(s, 0.55, yy, 1.5, 0.3, area, size=10.5, color=col, bold=True)
    txt(s, 2.1, yy, 4.85, 0.56, what, size=10.2, color=INK, spacing=1.0)
    txt(s, 7.15, yy, 5.63, 0.56, why, size=10.2, color=GREY, spacing=1.0)
    line(s, 0.55, yy+0.55, 12.78, yy+0.55, color=LGREY, lw=0.5)
    yy += 0.615
note(s, """
Do not soften anything here, and do not let the tone become apologetic either. The correct
register is: I know exactly what this project does and does not establish.

The first three are the ones that could sink a defence if an examiner raised them before I
did, so I raise them. One: there is no clean held-out test set — the reported one collapsed
and I am reporting on validation rows that also drove early stopping, which makes the numbers
optimistic. Two: one seed per variant, and my measured run-to-run noise is 4.5 W/m2, which is
the same order as several of the differences I discuss. Three: the evaluation window is a
single monsoon season.

If asked "so which of your conclusions survive all that?" — I have a prepared answer. The
skill-against-persistence result survives: it is large, consistent across every model and
every horizon, and robust to the eight percent optimism from the overnight-gap defect. The
direction of the imagery result survives: it is consistent across four architectures, three
horizons, two sample sizes and a paired significance test. What does not survive is any
precise magnitude — I will not defend "the image branch costs 3.9 W/m2", only "the image
branch does not help in this data regime."

The clear-sky convention item is worth dwelling on if there is time, because it is a genuine
physics bug and finding it required looking at the data rather than the code. The maximum
observed clear-sky index should cap near one at every hour. Instead it drifts from 0.69 in the
morning to 1.43 in the late afternoon. That monotone drift is the fingerprint of dividing an
hourly-mean irradiance by an instantaneous clear-sky value: with the sun rising, the
instantaneous value exceeds the hourly mean and the ratio is suppressed; with the sun setting,
the reverse. The corrected hour-mean function exists in the codebase and is used for the
deployment cap, but the dataset column was never rebuilt with it — so the clear-sky index
feature and the gate's first input both carry the bias. Fixing it means rebuilding the dataset
and retraining everything, which is exactly why it is future work rather than a quick patch.
""")

# ═══════════════════════════════════ 20 · CONTRIBUTIONS ══════════════════════
s = S("Contributions, conclusions and what I would do next", "Closing", rule=TEAL)
txt(s, 0.55, 1.4, 6.05, 0.26, "CONTRIBUTIONS", size=10.5, color=ACC, bold=True)
contrib = [
 ("Engineering", "A complete, automated pipeline from three public feeds to a verified daily "
  "forecast, with one shared model definition across training and inference, validated "
  "satellite ingestion, config sidecars and diagnostics in every output."),
 ("Implementation", "A physics-gated dual-branch fusion architecture with 8-head "
  "cross-attention over 196 patches, an optical-flow advection prior, SWIMSEG-pretrained "
  "encoders and a heteroscedastic 3-horizon head."),
 ("Experimental", "Six trained variants and five baselines scored on identical rows, with "
  "skill scores, paired Diebold–Mariano tests, CRPS and interval width — a comparison the "
  "original results could not support."),
 ("Analytical", "A mechanistic explanation of the negative result: the gate learned the "
  "physically correct direction but only 13 % of its available range, and 92.9 % of the "
  "parameters sit in a branch that adds variance, not signal."),
 ("Methodological", "Self-audit as a deliverable: an 18-sample evaluation collapse, a "
  "clear-sky timestamp-convention error, an overnight-gap target defect and four train/serve "
  "divergences, each located, quantified and reported."),
]
yy = 1.72
for k, v in contrib:
    txt(s, 0.55, yy, 1.35, 0.3, k, size=10, color=NAVY, bold=True)
    txt(s, 1.98, yy, 4.62, 0.72, v, size=10, color=INK, spacing=1.0)
    yy += 0.74
txt(s, 0.55, 5.5, 6.05, 0.26, "NOT CLAIMED", size=10.5, color=RED, bold=True)
txt(s, 0.55, 5.8, 6.05, 1.1,
    "No new algorithm, layer or loss. Every component is standard; the assembly, the "
    "evaluation protocol and the deployment are the work. No claim of state of the art, and "
    "no claim that satellite fusion fails in general — only that it does not pay for itself "
    "in this data regime.", size=10.5, color=GREY, spacing=1.05)
txt(s, 6.95, 1.4, 5.83, 0.26, "CONCLUSIONS", size=10.5, color=ACC, bold=True)
concl = [
 ("1", "Learned models beat the operational reference decisively, and by more as the horizon "
       "grows: +20 % skill at t+1 h to +38 % at t+3 h against smart persistence."),
 ("2", "Satellite imagery did not improve this forecaster. Removing the image branch is "
       "significantly better at t+1 h (p = 0.0001) and t+2 h (p = 0.019)."),
 ("3", "The physics gate did not beat naive concatenation, because it collapsed to a "
       "near-constant 0.49 blend — directionally correct, operationally inert."),
 ("4", "Gradient-boosted trees on 14 tabular features are the strongest model here, "
       "significantly at t+1 h and indistinguishable beyond it."),
 ("5", "The probabilistic head is well calibrated without recalibration: 90–92 % coverage "
       "at a nominal 90 %, though the intervals are wide."),
 ("6", "Evaluation design dominated architecture: sample size changed the ranking more than "
       "any architectural choice did."),
]
yy = 1.72
for k, v in concl:
    tag(s, 6.95, yy, 0.32, 0.3, k, color=WHITE, fill=NAVY, size=9.5)
    txt(s, 7.42, yy+0.02, 5.36, 0.6, v, size=10.3, color=INK, spacing=1.02)
    yy += 0.63
txt(s, 6.95, 5.52, 5.83, 0.26, "FUTURE WORK, IN PRIORITY ORDER", size=10.5, color=ACC, bold=True)
fut = [("1", "Fix the split (filter imaged rows first, assert the size), retrain, and report "
             "a genuine ~1,100-sample two-season test."),
       ("2", "Three to five seeds per variant; report mean ± sd.  ≈6 GPU-hours."),
       ("3", "Rebuild the dataset on a continuous hourly index with hour-mean clear sky, "
             "then retrain once, deliberately."),
       ("4", "Close divergences 1–2, run the corrected deployment for a fortnight, publish "
             "a before/after table."),
       ("5", "Frozen encoder + light probe; a gate that invokes imagery only when k_t < 0.3.")]
yy = 5.80
for k, v in fut:
    txt(s, 6.95, yy, 0.25, 0.24, k+".", size=9.4, color=ACC, bold=True)
    txt(s, 7.24, yy, 5.54, 0.4, v, size=9.4, color=INK, spacing=0.98)
    yy += 0.235
note(s, """
Close by separating what I built from what I discovered, and by being precise about novelty.

On contributions, the honest framing is: no new algorithm. Every component — BiLSTM with
attention pooling, EfficientNet encoders, multi-head cross-attention, a gating MLP, a Gaussian
NLL head — is standard and cited. What is mine is the assembly, the physical conditioning of
the gate, the evaluation protocol, the deployment, and the analysis of why the assembly did
not work. If an examiner asks "what is novel here?", I answer that directly rather than
inflating it: the novelty is modest and lies in the integration and the rigour, and the most
valuable output is a carefully established negative result plus a mechanistic explanation for
it.

Conclusion six is the one I would most want them to remember, because it is the transferable
lesson: the evaluation design mattered more than any architectural decision I made. Changing
the sample size from eighteen to nine hundred and fourteen reordered the model ranking
completely; no architectural change in the whole grid moved performance nearly as much.

On future work, be concrete about cost so it reads as a plan rather than a wish list. Fixing
the split is one line plus about an hour of GPU. Seed replication is six GPU-hours. The dataset
rebuild with the corrected clear-sky convention is the expensive one because it invalidates
every number and requires a full retrain, which is exactly why it should be done once and
deliberately rather than piecemeal.

If asked the classic closing question — what would you do with another six months — my answer
is: not a bigger model. I would fix the evaluation, add seeds, rebuild the dataset on a
continuous hourly index, and then test the one hypothesis this project actually generated,
which is that imagery helps only in the overcast regime and should be gated on that regime
rather than blended in everywhere.
""")

# ═══════════════════════════════════════ REFERENCES ══════════════════════════
def A(title, kicker):
    sl = slide_base(prs, title, kicker, None, NAVY)
    return sl

s = A("References", "Appendix")
refs = [
 ("Data and imagery", [
  "NICT Science Cloud. Himawari-8/9 real-time web imagery, tile service D531106/4d/550. "
  "National Institute of Information and Communications Technology, Japan.",
  "JAXA P-Tree. Himawari Standard Data distribution service (fallback source).",
  "Open-Meteo. Historical Weather API (ERA5 reanalysis) and Forecast API. open-meteo.com.",
  "European Commission JRC. PVGIS 5.3 photovoltaic geographical information system.",
  "Meteorological Service Singapore / data.gov.sg. Real-time weather readings API."]),
 ("Methods used in the implementation", [
  "Dev, S., Lee, Y. H., Winkler, S. (2017). Color-based segmentation of sky/cloud images "
  "from ground-based cameras. IEEE J. Sel. Topics Appl. Earth Obs. Remote Sensing, 10(1). "
  "— SWIMSEG dataset used for encoder pretraining.",
  "Tan, M., Le, Q. (2019). EfficientNet: rethinking model scaling for convolutional neural "
  "networks. ICML. — B0/B2 backbones via timm.",
  "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS. — multi-head "
  "cross-attention.",
  "Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).",
  "Nix, D. A., Weigend, A. S. (1994). Estimating the mean and variance of the target "
  "probability distribution. ICNN. — heteroscedastic Gaussian NLL.",
  "Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion. SCIA. "
  "— optical flow, via OpenCV.",
  "Ineichen, P., Perez, R. (2002). A new airmass independent formulation for the Linke "
  "turbidity coefficient. Solar Energy, 73(3). — clear-sky model, via pvlib.",
  "Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. ICLR. — AdamW.",
  "Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.",
  "Breiman, L. (2001). Random forests. Machine Learning, 45(1)."]),
 ("Evaluation methodology", [
  "Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy. J. Business & "
  "Economic Statistics, 13(3).",
  "Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the equality of prediction mean "
  "squared errors. Int. J. Forecasting, 13(2). — small-sample correction applied on slide 13.",
  "Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and "
  "estimation. JASA, 102(477). — CRPS.",
  "Marquez, R., Coimbra, C. F. M. (2013). Proposed metric for evaluation of solar "
  "forecasting models. J. Solar Energy Engineering, 135(1). — skill score conventions."]),
 ("Software", [
  "PyTorch 2.x · timm · pvlib-python 0.15 · scikit-learn 1.8 · LightGBM 4.6 · OpenCV 5.0 · "
  "pandas · NumPy · Matplotlib · python-pptx."]),
]
col = [(0.55, 6.0), (6.95, 5.83)]
ci, yy = 0, 1.42
for head, items in refs:
    x, w = col[ci]
    if yy > 6.2 and ci == 0:
        ci, yy = 1, 1.42
        x, w = col[ci]
    txt(s, x, yy, w, 0.24, head.upper(), size=10, color=ACC, bold=True)
    yy += 0.26
    for it in items:
        nl = max(1, -(-len(it) // int(w * 72 / (9.2 * 0.5))))
        txt(s, x + 0.16, yy, w - 0.16, 0.3, "·  " + it, size=9.2, color=INK, spacing=1.0)
        yy += 0.155 * nl + 0.055
        if yy > 6.55 and ci == 0:
            ci, yy = 1, 1.42
            x, w = col[ci]
    yy += 0.12
note(s, """
Only cite what the implementation actually uses. Every method reference on this slide
corresponds to a component that is in the code: timm's EfficientNet, PyTorch's
MultiheadAttention and LSTM, OpenCV's Farneback implementation, pvlib's Ineichen clear-sky
model, LightGBM and scikit-learn's random forest, AdamW.

Be ready to distinguish two categories if asked. External background: the papers that define
the methods I used. Project implementation: how they are wired together in model.py. I do not
claim to have implemented the papers — I used the library implementations, and the SWIMSEG
pretraining stage is the one place where I built something on top of a published dataset
rather than just calling a library.

The evaluation references matter more than they look. Diebold-Mariano with the
Harvey-Leybourne-Newbold correction is what licenses the significance claims on slide 13, and
Marquez and Coimbra is the standard reference for how skill scores should be reported in solar
forecasting — that is why skill against smart persistence, not against a naive mean, is the
headline metric throughout.
""")

# ═══════════════════════════════════ A1 · TRACEABILITY ═══════════════════════
s = A("A1 · Code-to-slide traceability", "Appendix")
rows = [
 ["4, 6", "Architecture, tensor shapes, ablations", "model.py:277-399",
  "PhysicsGatedFusionModelV2 · forward, _image_patches"],
 ["6, 7", "BiLSTM + temporal attention pooling", "model.py:56-80",
  "TemporalSelfAttention, BiLSTMEncoder"],
 ["6, 7", "8-head cross-attention over 196 patches", "model.py:256-274",
  "MultiHeadCrossAttention"],
 ["6, 7, 15", "Physics gate (193 params)", "model.py:326-329", "self.gate (Linear 4→D/8→1, σ)"],
 ["5, 7", "Clear-sky index, engineered features", "solar_pv_main.ipynb cell 4",
  "clearsky_ratio, sin/cos hour & month"],
 ["5", "Dataset assembly, ±60 min image match", "combined_dataset.py:61-95, 100-207",
  "match_images, build_combined_dataset"],
 ["5", "Weather + clear-sky fetch", "historical_data.py:13-70", "HistoricalDataCollector.fetch_data"],
 ["5", "PV reference series", "fetch_solcast.py", "PVGIS + pvlib simulation"],
 ["5, 6", "Image tensor cache (PNG → grayscale .npy)", "solar_pv_main.ipynb cells 8-9, 16",
  "save_images_as_npy, npy_cache"],
 ["5, 7", "Optical flow between hourly frames", "solar_pv_main.ipynb cell 17",
  "cv2.calcOpticalFlowFarneback, FLOW_SIZE 64"],
 ["5, 9, 10", "Split, windows, valid_indices", "solar_pv_main.ipynb cell 19",
  "GHIForecastDataset, 70/15/15 positional"],
 ["3, 6", "SWIMSEG pretraining stage", "solar_pv_main.ipynb cells 11-14",
  "SwimSegDataset, _FPNDecoder, SwimSegModel"],
 ["7, 8", "Gaussian NLL, two-phase schedule", "solar_pv_main.ipynb cell 23",
  "gaussian_nll_loss, train_model, unfreeze at epoch 20"],
 ["9, 11", "Experiment grid + evaluation", "solar_pv_main.ipynb cell 25",
  "EXPERIMENTS, evaluate_model, baseline_results"],
 ["9, 11, 12", "Tabular baselines", "solar_pv_main.ipynb cell 6",
  "BASELINE_FEATURES, LinearRegression / RF / LightGBM"],
 ["1, 7, 12", "Persistence baselines and skill", "model.py:669-704",
  "persistence_forecast, smart_persistence_forecast, skill_score"],
 ["17", "Validated satellite ingestion", "current_data.py:105-191",
  "validate_tile, fetch_image, fetch_frame_series"],
 ["17", "Lookback window + gate features at serve time", "model.py:538-573",
  "build_lookback_window, compute_gate_features"],
 ["17", "Live forecast, physics cap, diagnostics", "predict.py:131-280",
  "apply_physics_cap, predict"],
 ["17", "Verification loop", "verify.py:56-272", "fetch_actual_ghi, main"],
 ["6, 17, 18", "Checkpoint save/load with config sidecar", "model.py:186-235",
  "save_checkpoint, load_model"],
]
table(s, 0.55, 1.42, 12.23,
      ["Slide", "Claim / component", "Repository location", "Class · function"],
      rows, col_w=[0.9,3.9,3.2,4.2], size=8.7, row_h=0.222, head_h=0.27)
txt(s, 0.55, 6.42, 12.23, 0.5,
    "Line numbers refer to the state of the repository at commit 56b7371. Where a claim in "
    "this deck came from an analysis performed during preparation rather than from a committed "
    "artefact, it is marked as such on the slide and detailed in A4.",
    size=9.5, color=GREY, italic=True)
note(s, """
This appendix exists so that any technical claim in the deck can be traced to a file, a class
and a function in under ten seconds. If an examiner asks "where is that in the code?", turn to
this slide rather than describing from memory.

The rows most likely to be probed: model.py lines 262 to 399 for the architecture and the
ablation switches; cell 19 of the notebook for the dataset class, where valid_indices is
built and where the split defect on slide 10 originates; cell 23 for the training loop and the
epoch-20 unfreeze; cell 25 for the experiment grid.

The last row is worth knowing cold. save_checkpoint writes a .config.json sidecar alongside
every checkpoint because the number of attention heads cannot be recovered from tensor shapes —
a model loaded with the wrong head count produces silently wrong predictions rather than an
error. That is the kind of failure mode that never shows up in a validation metric, which is
why it is handled explicitly.
""")

# ═══════════════════════════════════ A2 · Q&A 1 ══════════════════════════════
def qa_slide(title, pairs):
    """Lays out Q/A pairs, sizing each block from the answer's wrapped line count."""
    sl = A(title, "Appendix · viva preparation")
    CPL = 168          # characters per line at 10.1 pt across 11.9 in
    heights = [0.27 + 0.185 * max(1, -(-len(a) // CPL)) + 0.13 for _, a in pairs]
    total = sum(heights)
    avail = 6.92 - 1.40
    scale = min(1.0, avail / total)
    yy = 1.40
    for (q, a), hgt in zip(pairs, heights):
        txt(sl, 0.55, yy, 12.23, 0.26, "Q   " + q, size=11.3, color=NAVY, bold=True)
        txt(sl, 0.9, yy + 0.26, 11.9, hgt - 0.3, a, size=10.1, color=INK, spacing=1.0)
        yy += hgt * scale
        line(sl, 0.55, yy - 0.07, 12.78, yy - 0.07, color=LGREY, lw=0.5)
    if total > avail:
        print(f"  ! qa_slide over by {total-avail:.2f} in: {title}")
    return sl

s = qa_slide("A2 · Anticipated questions — architecture and data", [
 ("Why EfficientNet-B2 rather than a ResNet or a vision transformer?",
  "Parameter efficiency at 224×224 with ~6.7 k training samples. B2 gives 352-channel 7×7 maps "
  "at 7.2 M parameters; a ViT would need far more data or heavier pretraining. The results "
  "show even B2 was too much capacity — the small preset used B0, but shrinking the whole "
  "network also cost the temporal branch 23 W/m²."),
 ("Why is the attention query the BiLSTM state rather than a learned token?",
  "So the image pooling is conditioned on the recent weather trajectory — the temporal state "
  "asks the image what to look at. The cost is that the image branch can never be evaluated "
  "independently, which is exactly why the ablation named “CNN-only” in the code is renamed "
  "“image + temporal query” in this deck."),
 ("What are the tensor shapes through the image branch?",
  "(B,9,224,224) splits into three (B,3,224,224) frames → global_enc → (B,352,7,7) each; the "
  "RoI crop (B,3,112,112) is bilinearly upsampled to 224 → roi_enc → (B,352,7,7). Each map "
  "flattens to 49 tokens, giving 4 × 49 = 196 patch tokens of 352 channels → projected to 256."),
 ("Is future_clearsky leakage?",
  "No. Clear-sky GHI is a deterministic function of latitude, longitude and time — the "
  "Ineichen model with a turbidity climatology. It contains no information about the weather "
  "at t+h and is exactly as available at issue time as the calendar is. The genuine leak "
  "risk was pv_actual (r = 0.9999 with GHI); it is excluded from every feature set."),
 ("Is there data leakage across the split?",
  "No. The split is chronological and never shuffled; each split builds its own windows "
  "inside its own rows (valid_indices starts at index 23), so no validation window reaches "
  "back into training data — the first 23 rows of each split are consumed as history instead. "
  "Normalisation statistics come from the training rows only, and pv_actual is excluded "
  "everywhere. The residual contamination is model selection, not features: early stopping "
  "used these rows."),
 ("How representative is 2.5 years of one equatorial site?",
  "It covers three full monsoon cycles, which is adequate for seasonality but thin for rare "
  "events. The bigger problem is the evaluation window, which is a single north-east monsoon "
  "season — that is on the limitations slide."),
 ("Why grayscale images rather than the RGB composite?",
  "For consistency with the JAXA fallback source, which is single-channel. The tensor keeps "
  "three identical channels so ImageNet-pretrained weights still apply. The cost is any "
  "colour-based cloud-type discrimination that the RGB composite would carry."),
])
note(s, """
Rehearse these until the answers are one breath each; a hesitant answer on the architecture
reads worse than a wrong number.

Two of these deserve extra care.

The leakage question, because examiners always ask it and because there is a real partial
answer here. Future clear-sky is not leakage: it is solar geometry, computable years in
advance, and every operational solar forecast uses it. But I volunteer the genuine caveat
rather than claiming perfection: the 24-step lookback window at the very start of the
validation split reaches back across the boundary into training rows. That affects the first
twenty-four validation samples — 2.6 percent of the split — and I would fix it with a
guard band of twenty-four rows at each boundary. Volunteering that buys far more credibility
than a flat "no leakage."

The query-design question, because it exposes an architectural consequence I had to correct.
Making the BiLSTM state the attention query means the image branch cannot be evaluated
independently — there is no configuration of this model that uses imagery alone. That is why
the ablation labelled CNN-only in the code is mislabelled, and why I renamed it throughout the
deck. If I wanted a true image-only variant I would pool the patches with a learned constant
query.
""")

# ═══════════════════════════════════ A3 · Q&A 2 ══════════════════════════════
s = qa_slide("A3 · Anticipated questions — training, evaluation and results", [
 ("Why Gaussian NLL rather than MSE or pinball loss?",
  "The deliverable is an interval, not a point. NLL learns μ and σ jointly and gives a "
  "closed-form CRPS. Pinball loss on quantiles would be the natural alternative and avoids "
  "the Gaussian assumption — worth testing, since irradiance residuals are skewed near the "
  "clear-sky ceiling."),
 ("How were the hyperparameters chosen?",
  "By hand, guided by validation NLL, not by a search. lr 3 × 10⁻⁵ because higher rates "
  "destabilised the σ head under fp16; weight decay 0.1 and dropout 0.15 as a response to "
  "the visible overfitting; batch 128 to fill the GPU. No Bayesian or grid search was run — "
  "that is a limitation, not a design choice."),
 ("How did you prevent overfitting, and did it work?",
  "Frozen encoders for 20 epochs, weight decay 0.1, dropout, gradient clipping, early stopping "
  "on validation NLL. Partially: the curves on slide 8 show validation loss turning upward "
  "four epochs after the CNN is unfrozen. Early stopping caught it, but the underlying "
  "capacity mismatch remained."),
 ("Why report the validation split rather than the test split?",
  "Because the test split contains 18 imaged samples from two days — slide 10. The validation "
  "rows are 914 samples over 93 days. They are contaminated by early stopping and I say so "
  "everywhere; a contaminated estimate on 914 samples is far more informative than a clean "
  "one on 18."),
 ("Is the comparison with LightGBM fair?",
  "Yes, and deliberately generous to LightGBM: it receives 14 features at time t including "
  "three GHI lags, trained on the same rows and scored on identical samples. The deep models "
  "additionally receive 24-step history and four image tensors. If anything the deep models "
  "have the information advantage."),
 ("Why does the small model perform so much worse?",
  "It shrinks everything at once — B0 backbone, 128-d fusion, 4 heads, 64-unit LSTM, dropout "
  "0.3. The 23 W/m² loss says the temporal branch needed the capacity even though the image "
  "branch did not. It is evidence against a naive “just make it smaller” fix, and it is why "
  "the future-work item is a frozen encoder with a light probe rather than uniform shrinkage."),
 ("Which single result is the most important?",
  "The skill curve on slide 12: +20 % to +38 % against smart persistence, rising with horizon. "
  "It is the only result that is large, physically explicable, consistent across every model, "
  "and robust to the defects on the limitations slide."),
])
note(s, """
The two questions here that can turn hostile are the validation-split question and the
fairness question. Answer both with the same posture: acknowledge the weakness precisely, then
show the reasoning.

On reporting validation rather than test: do not be defensive. The choice is between a clean
estimate on eighteen samples and a contaminated estimate on nine hundred and fourteen. An
eighteen-sample MAE has a confidence interval so wide it cannot distinguish any two models in
the grid — I can show that the ranking literally inverts. The contamination in the validation
set is early stopping and checkpoint selection, which biases the absolute numbers optimistically
but applies equally to every deep variant, so the comparison between them remains meaningful.
And note it does not apply at all to the tabular baselines, which never saw those rows in any
form — if anything that biases the comparison against my own deep models, which makes the
LightGBM result conservative.

On fairness to LightGBM: the honest position is that the comparison favours the deep models on
information. LightGBM gets fourteen features at a single time step. The deep models get
twenty-four hours of history plus four image tensors plus the same clear-sky information. They
still lose at t+1h. That is not an unfair comparison; it is a decisive one.

On hyperparameters, do not pretend there was a search. There was not. Say it plainly and name
it as a limitation.
""")

# ═══════════════════════════════════ A4 · REPRODUCTION ═══════════════════════
s = A("A4 · Reproduction evidence and code-vs-documentation discrepancies", "Appendix")
txt(s, 0.55, 1.4, 6.1, 0.26, "INDEPENDENT REPRODUCTION PERFORMED FOR THIS DECK",
    size=10, color=ACC, bold=True)
txt(s, 0.55, 1.68, 6.1, 0.5,
    "The notebook pipeline was re-implemented from the committed CSV and checkpoints and run "
    "in fp32 on CPU/MPS. Agreement with the committed results:", size=10, color=INK, spacing=1.03)
table(s, 0.55, 2.25, 6.1,
      ["Quantity", "committed", "reproduced"],
      [["clean rows after dropna", "9,574", "9,574 ✓"],
       ["train / val / test rows", "6,701 / 1,436 / 1,437", "identical ✓"],
       ["deep samples per split", "6,675 / 914 / 18", "identical ✓"],
       ["GHI mean, std (train)", "476.89 / 258.44", "identical ✓"],
       ["11 normalisation means/stds", "train_stats.json", "match to 1e-3 ✓"],
       ["MAE — naive concat (n=18)", "133.09", "132.98"],
       ["MAE — no ghi_lag1 (n=18)", "138.30", "138.40"],
       ["MAE — image + query (n=18)", "142.58", "142.40"],
       ["MAE — LSTM-only (n=18)", "147.27", "147.29"],
       ["MAE — physics-gated (n=18)", "148.37", "148.17"],
       ["MAE — small preset (n=18)", "262.93", "262.21"]],
      col_w=[2.6,1.8,1.7], size=9.2, row_h=0.235, head_h=0.27)
txt(s, 0.55, 5.15, 6.1, 0.75,
    "Residual differences < 1 % are attributable to fp16 autocast at evaluation time in the "
    "notebook versus fp32 here. One trap worth recording: the cached .npy tensors must be the "
    "grayscale-converted version; using the older RGB cache changes MAE by up to 46 W/m².",
    size=9.5, color=GREY, spacing=1.03, italic=True)
txt(s, 0.55, 6.0, 6.1, 0.26, "NEW ANALYSES ADDED FOR THIS DECK", size=10, color=ACC, bold=True)
txt(s, 0.55, 6.28, 6.1, 0.7,
    "914-sample re-evaluation of all six checkpoints · tabular baselines rescored on identical "
    "rows · Diebold–Mariano tests · CRPS and interval width · gate-behaviour statistics · "
    "error stratification · same-day-only recomputation · CPU latency measurement.",
    size=9.5, color=INK, spacing=1.03)
txt(s, 6.95, 1.4, 5.83, 0.26, "WHERE CODE AND DOCUMENTATION DISAGREE  (code wins)",
    size=10, color=ACC, bold=True)
disc = [
 ("model.py:9-19 docstring", "Calls v1 “DEPLOYED” and v2 “experimental, no trained "
  "checkpoint yet”. All six checkpoints contain global_enc.* keys — every one is v2, and v2 "
  "is what predict.py loads. The docstring is stale."),
 ("Ablation name “cnn”", "Uses H_a, whose attention query is the BiLSTM output, so the "
  "temporal branch is still trained and still drives pooling. Renamed “image + temporal "
  "query” throughout this deck."),
 ("verify.py header", "Describes the Open-Meteo shortwave radiation series as “measured”. "
  "It is the same reanalysis product used to build the training targets — analysis, not an "
  "independent instrument."),
 ("compute_clearsky_hour_mean", "The corrected hour-mean clear-sky function exists and is "
  "used by the deployment cap, but the dataset's ghi_clearsky column and therefore "
  "clearsky_ratio still use the instantaneous value."),
 ("predict.py docstring", "States that live station weather feeds the model. It is fetched, "
  "printed and written to the output JSON, but never enters the input tensor."),
 ("results/*.csv", "The two result files were produced on different evaluation sets "
  "(18 vs 1,437 rows) and were never comparable — see slide 10."),
]
yy = 1.72
for k, v in disc:
    txt(s, 6.95, yy, 2.3, 0.3, k, size=9.2, color=NAVY, bold=True, font="Consolas", spacing=1.0)
    txt(s, 9.35, yy, 3.43, 0.75, v, size=9.2, color=INK, spacing=1.0)
    yy += 0.82
note(s, """
This appendix is the answer to "how do I know any of this is true?" and to "did you check
your own documentation against your own code?".

On reproduction, state the method before the result: I re-implemented the notebook's data
pipeline independently from the committed CSV and checkpoints, ran it in float32, and compared.
The split sizes came out identical — 6,675, 914 and 18 — the normalisation statistics matched
train_stats.json, and all six test MAEs landed within one percent of the committed values. The
gap is fp16 autocast in the original evaluation versus fp32 in mine.

The trap in the middle of that slide is worth mentioning aloud because it is a good story
about silent failure. My first reproduction attempt gave the weather-only model exactly right
and every image-using model wrong by up to forty-six W/m2. The reason: the notebook converts
its cached image tensors to grayscale, but the cache on this machine still held the older RGB
version. The weather-only variant matched because it never touches the image branch — which is
how I localised the problem in one step. It is a small illustration of the same theme as slide
17: image inputs fail silently, and only a differential test catches it.

The right-hand column is deliberate. I went looking for places where my own comments and
docstrings disagree with my own code, and I found six. The rule I applied is that the
implementation is the source of truth and the documentation is the thing that is wrong. The
most important one is the stale docstring claiming version one is deployed when all six
checkpoints are version two. None of these change a result; all of them would embarrass me if
an examiner found them first.
""")

# ═══════════════════════════════════ A5 · CLEAR-SKY ══════════════════════════
s = A("A5 · The clear-sky timestamp-convention finding", "Appendix · physics")
pic(s, 0.55, 1.45, FIG+'f10_clearsky.png', w=6.1)
caption(s, 0.55, 5.05, 6.1,
        "Maximum observed clear-sky index by hour over all 9,580 rows. A physically "
        "meaningful k_t should cap near 1.0–1.1 at every hour.")
bullets(s, 7.05, 1.45, 5.73, 5.4, [
 (0, [("The symptom.  ", {'bold':True,'color':NAVY}),
      ("Maximum k_t drifts monotonically from 0.69 at 08:00 to 1.43 at 17:00. 425 rows "
       "exceed 1.1. A cloud-enhancement excursion would be scattered across the day; a "
       "monotone morning-to-evening ramp is systematic.", {})]),
 (0, [("The cause.  ", {'bold':True,'color':NAVY}),
      ("Open-Meteo labels each hourly irradiance with the END of its averaging window — the "
       "14:00 value is the mean over 13:00→14:00. pvlib's get_clearsky returns the "
       "INSTANTANEOUS value at 14:00. With the sun rising the instantaneous value exceeds "
       "the hourly mean, so k_t is suppressed; with the sun setting the reverse. Hence the ramp.", {})]),
 (0, [("Where it propagates.  ", {'bold':True,'color':NAVY}),
      ("clearsky_ratio is tabular feature #1 and gate input #1; ghi_clearsky is a tabular "
       "feature; future_clearsky enters the enrich layer; and the smart-persistence baseline "
       "is built from the same ratio. The bias touches models and baselines alike.", {})]),
 (0, [("The fix, already in the codebase.  ", {'bold':True,'color':TEAL}),
      ("compute_clearsky_hour_mean() averages pvlib's clear sky over the preceding hour at "
       "10-minute steps — the matching convention. It is applied to the deployment physics "
       "cap but not to the dataset column, so the training features were never rebuilt with it.", {})]),
 (0, [("Why it is future work rather than a patch.  ", {'bold':True,'color':ACC}),
      ("Correcting the column changes every feature, every target-adjacent quantity and every "
       "baseline, so it requires a dataset rebuild and a full retrain. It should be done once, "
       "together with the continuous-hourly-index fix, and reported as a v1-versus-v2 "
       "comparison.", {})]),
], size=11.5)
note(s, """
Bring this appendix out if anyone asks about the clear-sky model, about feature engineering,
or about what I found that a code review would not.

This one was found by interrogating the data, not the code. The clear-sky index is bounded
above by roughly one — a little above one under cloud-edge enhancement, which is a real
physical effect. So I tabulated the maximum observed value by hour, expecting a flat line just
above one. Instead it ramps monotonically from 0.69 in the morning to 1.43 in the late
afternoon. Monotone structure like that is never a physical effect; it is a convention error.

The explanation is a timestamp-convention mismatch. Open-Meteo stamps each hourly average with
the end of its window, so the value at 14:00 is the mean over the preceding hour. pvlib's
clear-sky function returns the instantaneous value at 14:00. In the morning, when the sun is
climbing, the instantaneous value is larger than the preceding-hour mean, so the ratio is
pushed down. In the afternoon it is smaller, so the ratio is pushed up. That is exactly the
shape in the figure.

What makes this worth a slide is where it propagates. The clear-sky index is tabular feature
number one and gate input number one. The clear-sky value itself is another feature, and the
future clear-sky values feed the enrich layer. The smart-persistence baseline is built from
the same ratio. So the bias is everywhere — though, importantly, it is in the baseline too,
which is part of why the skill comparison is more robust than the absolute MAEs.

And the sting: the correct function already exists in the codebase. It was written to fix the
deployment physics cap, where the error was most visible near sunrise and sunset, but the
dataset column was never rebuilt with it. Fixing it properly means rebuilding the dataset and
retraining everything, which is why it sits in future work as a deliberate one-time change
rather than a quick patch.
""")

prs.save('deck/GHI_Forecasting_Defence.pptx')
print('TOTAL slides:', len(prs.slides.__iter__.__self__._sldIdLst))
