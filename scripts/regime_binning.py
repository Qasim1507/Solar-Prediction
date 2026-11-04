"""Regime decomposition of the backtest under three binnings.

`tab_regime` bins by the clear-sky index at the *target* hour, which conditions on
the outcome and biases the extreme bins against any mean-reverting forecast. This
script reports that binning alongside two others -- by what the model predicted,
and by what the sky looked like when the forecast was issued -- with a paired
bootstrap CI on the MAE difference against smart persistence.

    python3 scripts/regime_binning.py [path-to-verification-log]
"""
import csv, sys, random

random.seed(0)
LOG = sys.argv[1] if len(sys.argv) > 1 else "verification_log_backtest.csv"

# The log also carries standalone smart-persistence rows (method smart_persistence,
# horizon t+1h-sp) from the A/B trial; those are not v3 forecasts.
rows = [r for r in csv.DictReader(open(LOG)) if r["method"] == "v3_ensemble"]
for r in rows:
    cs = float(r["clearsky_wm2"])
    r["ae"] = float(r["abs_error_wm2"])
    r["spae"] = float(r["sp_abs_error_wm2"])
    r["kt_issue"] = float(r["sp_forecast_wm2"]) / cs if cs > 0 else float("nan")
    r["cov"] = r["in_90ci"] == "True"

BINS = [("Overcast   (kt<0.35)", lambda v: v < 0.35),
        ("Cloudy     (0.35-0.65)", lambda v: 0.35 <= v < 0.65),
        ("Broken     (0.65-0.85)", lambda v: 0.65 <= v < 0.85),
        ("Near-clear (kt>=0.85)", lambda v: v >= 0.85)]


def bootstrap(sample, draws=20000):
    """Paired bootstrap on mean(model AE - smart-persistence AE)."""
    d = [x["ae"] - x["spae"] for x in sample]
    n = len(d)
    means = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n
                   for _ in range(draws))
    return sum(d) / n, means[int(0.025 * draws)], means[int(0.975 * draws)]


def report(title, key):
    print("=" * 76)
    print(title)
    print("=" * 76)
    for name, in_bin in BINS:
        sample = [r for r in rows if in_bin(key(r))]
        if not sample:
            continue
        n = len(sample)
        mae = sum(r["ae"] for r in sample) / n
        sp = sum(r["spae"] for r in sample) / n
        diff, lo, hi = bootstrap(sample)
        verdict = ("v3 better" if hi < 0 else
                   "SP better" if lo > 0 else "indistinguishable")
        print(f"{name:24s} n={n:5d}  v3 {mae:7.1f}  SP {sp:7.1f}  "
              f"skill {100 * (1 - mae / sp):+6.1f}%")
        print(f"{'':24s}  diff {diff:+7.1f} W/m2  95% CI [{lo:+.1f}, {hi:+.1f}]"
              f"  -> {verdict}")
        print(f"{'':24s}  PICP90 {100 * sum(r['cov'] for r in sample) / n:.1f}%")
    print()


report("A) by ACTUAL kt at target hour (as in tab_regime; conditions on outcome)",
       lambda r: float(r["kt_actual"]))
report("B) by FORECAST kt (conditions on what the model said)",
       lambda r: float(r["kt_forecast"]))
report("C) by kt at ISSUE time (what was knowable; the operational question)",
       lambda r: r["kt_issue"])

n = len(rows)
print(f"OVERALL PICP90 = {100 * sum(r['cov'] for r in rows) / n:.1f}%"
      f"  (nominal 90%)  n={n}")
