"""
webapp/app.py — a local front end for the daily v3 loop.

    .venv/bin/python webapp/app.py        # http://127.0.0.1:8050

The two buttons run the same commands as the manual daily run, unchanged:

    predict_v3.py --out forecast_latest_v3.json
    verify.py --forecast forecast_latest_v3.json --log verification_log_v3.csv --no-model-fallback

Nothing here forecasts or scores anything. The app streams the scripts' output
and reads what they write: forecast_latest_v3.json, datanow/, and
verification_log_v3.csv. Target-hour satellite images for the verification
view come from himawari_aws.extract_scan, the same NOAA crops the model sees.

After a successful run the app republishes the public site: it runs
export_static.py, commits site/ (and nothing else) and pushes main, which
Vercel redeploys. Set PUBLISH=0 to turn this off.
"""
import hmac
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

FORECAST = ROOT / "forecast_latest_v3.json"
LOG = ROOT / "verification_log_v3.csv"
BACKTEST_LOG = ROOT / "verification_log_backtest.csv"  # scripts/backtest.py
BACKTEST_FC = ROOT / "backtest" / "forecasts"           # one JSON per issue time
ARCHIVE = ROOT / "data" / "satellite_aws"               # the scans the model trained on
CSV = ROOT / "data" / "combined_dataset_v2.csv"
SUMMARY = ROOT / "datanow" / "current" / "current_summary.json"
SAT_DIR = ROOT / "datanow" / "satellite"
AWS_CACHE = SAT_DIR / "aws_cache"
RUNS = HERE / "runs"
# Fetch order in current_data.fetch_frame_series: t, t-10min, t-20min.
FRAMES = ["himawari_current.png", "himawari_prev1.png", "himawari_prev2.png"]
HORIZONS = ["t+1h", "t+2h", "t+3h"]

COMMANDS = {
    "predict": ["predict_v3.py", "--out", "forecast_latest_v3.json"],
    "verify": ["verify.py", "--forecast", "forecast_latest_v3.json",
               "--log", "verification_log_v3.csv", "--no-model-fallback"],
}

PUBLISH = os.environ.get("PUBLISH", "1") != "0"

# Public deployments set ADMIN_TOKEN; without it the app runs in local mode
# and anyone who can reach it may start jobs.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

sys.path.insert(0, str(ROOT))
app = Flask(__name__)


def is_admin():
    if not ADMIN_TOKEN:
        return True
    given = request.headers.get("X-Admin-Token", "")
    return hmac.compare_digest(given.encode(), ADMIN_TOKEN.encode())


# ── job runner ────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, kind):
        self.kind = kind
        self.state = "running"
        self.started = time.time()
        self.returncode = None
        self.lines = []


_job = None
_job_lock = threading.Lock()


def _stream(job, cmd):
    """Run cmd in ROOT, appending its output to the job log; return exit code."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            bufsize=1)
    for line in proc.stdout:
        job.lines.append(line.rstrip("\n"))
    return proc.wait()


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True)


def publish(job):
    """Export the static site, commit only site/, and push main to deploy it."""
    say = lambda msg: job.lines.append(f"[publish] {msg}")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        say(f"skipped: on branch {branch!r}, Vercel deploys main")
        return
    say("exporting static site...")
    if _stream(job, [str(PY), "-u", str(HERE / "export_static.py")]) != 0:
        say("export failed; site not updated")
        return
    _git("add", "--all", "site")
    if _git("diff", "--cached", "--quiet", "--", "site").returncode == 0:
        say("no changes to publish")
        return
    stamp = pd.Timestamp.now(tz="Asia/Singapore").strftime("%Y-%m-%d %H:%M")
    # The pathspec limits the commit to site/, whatever else is staged.
    r = _git("commit", "-m", f"site update {stamp} SGT ({job.kind})", "--", "site")
    if r.returncode != 0:
        say(f"commit failed: {(r.stderr or r.stdout).strip()}")
        return
    r = _git("push", "origin", "main")
    if r.returncode != 0:
        say(f"push failed (commit kept locally): {r.stderr.strip()}")
        return
    say("pushed; Vercel will redeploy in about a minute")


def _run(job):
    try:
        job.returncode = _stream(job, [str(PY), "-u", *COMMANDS[job.kind]])
    except Exception as e:
        job.lines.append(f"[webapp] failed to start: {e}")
        job.returncode = -1
    if job.returncode == 0 and job.kind == "predict":
        try:
            run_id = archive_run()
            job.lines.append(f"[webapp] archived run -> webapp/runs/{run_id}/")
        except Exception as e:
            job.lines.append(f"[webapp] could not archive run: {e}")
    if job.returncode == 0 and PUBLISH:
        try:
            publish(job)
        except Exception as e:
            job.lines.append(f"[publish] failed: {e}")
    job.state = "done" if job.returncode == 0 else "failed"


@app.post("/api/<kind>")
def start_job(kind):
    global _job
    if kind not in COMMANDS:
        abort(404)
    if not is_admin():
        return jsonify(error="runs are admin-only"), 403
    with _job_lock:
        if _job is not None and _job.state == "running":
            return jsonify(error=f"a {_job.kind} job is already running"), 409
        _job = Job(kind)
        threading.Thread(target=_run, args=(_job,), daemon=True).start()
    return jsonify(kind=kind, started=_job.started)


@app.get("/api/config")
def config():
    return jsonify(admin=is_admin())


@app.get("/api/job")
def job_status():
    if _job is None:
        return jsonify(kind=None, state="idle", lines=[], next=0)
    since = max(0, request.args.get("since", 0, type=int))
    lines = _job.lines[since:]
    return jsonify(kind=_job.kind, state=_job.state, started=_job.started,
                   returncode=_job.returncode, lines=lines,
                   next=since + len(lines))


# ── prediction view ───────────────────────────────────────────────────────────

def run_id_for(forecast_time):
    return pd.Timestamp(forecast_time).strftime("%Y%m%d_%H%M")


def archive_run():
    """Keep the forecast and its input frames; the next predict overwrites them."""
    fc = json.loads(FORECAST.read_text())
    run_id = run_id_for(fc["forecast_time_sgt"])
    dest = RUNS / run_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FORECAST, dest / "forecast.json")
    for name in FRAMES:
        if (SAT_DIR / name).exists():
            shutil.copyfile(SAT_DIR / name, dest / name)
    return run_id


@app.get("/api/frames")
def frames():
    out = []
    for name in FRAMES:
        p = SAT_DIR / name
        out.append({"name": name, "url": f"/img/live/{name}",
                    "mtime": p.stat().st_mtime if p.exists() else None})
    return jsonify(out)


@app.get("/img/live/<name>")
def live_image(name):
    if name not in FRAMES or not (SAT_DIR / name).exists():
        abort(404)
    return send_file(SAT_DIR / name, max_age=0)


@app.get("/api/forecast")
def forecast():
    fc = json.loads(FORECAST.read_text()) if FORECAST.exists() else None
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    return jsonify(forecast=fc,
                   weather=summary.get("weather_summary"),
                   collected=summary.get("collection_timestamp"))


# ── verification view ─────────────────────────────────────────────────────────

# key "YYYYmmdd_HHMM" (SGT) -> {"status": fetching|ready|missing, "path", "scan_utc"}
_images = {}
_images_lock = threading.Lock()


def _fetch_scan(key, t_sgt):
    """Target-hour crop from NOAA, stepping back 10 min like fetch_image_aws."""
    from himawari_aws import extract_scan
    t_utc = (pd.Timestamp(t_sgt).tz_localize("Asia/Singapore")
             .tz_convert("UTC").tz_localize(None).to_pydatetime())
    result = {"status": "missing", "path": None, "scan_utc": None}
    for attempt in range(4):
        t = t_utc - timedelta(minutes=10 * attempt)
        try:
            status, info = extract_scan(t, str(AWS_CACHE))
        except Exception as e:
            result["error"] = str(e)[:120]
            continue
        if status in ("ok", "skip"):
            result = {"status": "ready", "path": info,
                      "scan_utc": t.strftime("%Y-%m-%d %H:%M")}
            break
    result["at"] = time.time()
    with _images_lock:
        _images[key] = result


def scan_image(t_sgt):
    """Image record for an SGT time, starting a background fetch if needed."""
    key = pd.Timestamp(t_sgt).strftime("%Y%m%d_%H%M")
    with _images_lock:
        rec = _images.get(key)
        # NOAA posts scans ~15-20 min late, so retry a miss after a couple of minutes.
        stale_miss = (rec is not None and rec["status"] == "missing"
                      and time.time() - rec.get("at", 0) > 120)
        if rec is None or stale_miss:
            rec = _images[key] = {"status": "fetching"}
            threading.Thread(target=_fetch_scan, args=(key, t_sgt),
                             daemon=True).start()
    out = {"key": key, "status": rec["status"], "scan_utc": rec.get("scan_utc")}
    if rec["status"] == "ready":
        out["url"] = f"/img/scan/{key}"
    return out


@app.get("/img/scan/<key>")
def scan_file(key):
    with _images_lock:
        rec = _images.get(key)
    if not rec or rec.get("status") != "ready":
        abort(404)
    return send_file(rec["path"], max_age=3600)


@app.get("/img/run/<run_id>/<name>")
def run_image(run_id, name):
    p = RUNS / run_id / name
    if name not in FRAMES or not run_id.replace("_", "").isdigit() or not p.exists():
        abort(404)
    return send_file(p, max_age=3600)


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _read_log():
    if not LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG)
    if "method" not in df.columns:
        df["method"] = "v3_ensemble"
    df["method"] = df["method"].fillna("v3_ensemble")
    return df


def _forecast_for(ftime):
    """The forecast JSON for an issue time: the latest file, else the archive."""
    if FORECAST.exists():
        fc = json.loads(FORECAST.read_text())
        if fc["forecast_time_sgt"] == ftime:
            return fc
    p = RUNS / run_id_for(ftime) / "forecast.json"
    return json.loads(p.read_text()) if p.exists() else None


def _columns(ftime, fc, rows, image_for, now):
    """One column per horizon: forecast, log row and target-hour image."""
    by_h = {r["horizon"]: r for _, r in rows.iterrows()}
    fc_by_h = {f["horizon"]: f for f in (fc or {}).get("forecasts", [])}

    def entry(h):
        f, r = fc_by_h.get(h), by_h.get(h)
        if f is None and r is None:
            return None
        e = {"horizon": h,
             "method": (f or {}).get("method") or _clean(r["method"]),
             "forecast": (f or {}).get("ghi_forecast_wm2",
                                       _clean(r["forecast_wm2"]) if r is not None else None),
             "lower": (f or {}).get("ghi_lower_90"),
             "upper": (f or {}).get("ghi_upper_90"),
             "kt_forecast": (f or {}).get("kt_forecast",
                                          _clean(r["kt_forecast"]) if r is not None else None)}
        if r is not None:
            e.update(actual=_clean(r["actual_wm2"]),
                     abs_error=_clean(r["abs_error_wm2"]),
                     in_ci=bool(str(r["in_90ci"]).lower() == "true"),
                     kt_actual=_clean(r["kt_actual"]),
                     sp_forecast=_clean(r["sp_forecast_wm2"]),
                     sp_abs_error=_clean(r["sp_abs_error_wm2"]),
                     verified_at=_clean(r["verified_at"]))
        return e

    columns = []
    for i, h in enumerate(HORIZONS):
        f, r = fc_by_h.get(h), by_h.get(h)
        target = ((f or {}).get("time_sgt")
                  or (_clean(r["target_time"]) if r is not None else None)
                  or (pd.Timestamp(ftime) + timedelta(hours=i + 1))
                  .strftime("%Y-%m-%d %H:%M"))
        t = pd.Timestamp(target)
        if r is not None:
            status = "verified"
        elif t > now:
            status = "pending"
        else:
            status = "unverified"
        col = {"horizon": h, "target_time": target, "status": status,
               "v3": entry(h), "challenger": entry(f"{h}-sp")}
        col["image"] = image_for(t) if t <= now else {"status": "pending"}
        columns.append(col)
    return columns


@app.get("/api/verification")
def verification():
    log = _read_log()
    latest = (json.loads(FORECAST.read_text())["forecast_time_sgt"]
              if FORECAST.exists() else None)
    times = set(log["forecast_time"].astype(str)) if len(log) else set()
    if latest:
        times.add(latest)
    ftime = request.args.get("forecast_time") or latest
    if not ftime:
        return jsonify(error="no forecast yet"), 404

    fc = _forecast_for(ftime)
    if fc is not None:
        times.add(ftime)  # archived runs that have no log rows yet
    rows = log[log["forecast_time"].astype(str) == ftime] if len(log) else log
    now = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None)
    columns = _columns(ftime, fc, rows, scan_image, now)

    # What the model saw: the archived t frame if this run was archived,
    # else the NOAA scan at the issue time.
    run_id = run_id_for(ftime)
    if (RUNS / run_id / FRAMES[0]).exists():
        issue_img = {"status": "ready", "url": f"/img/run/{run_id}/{FRAMES[0]}",
                     "source": "archived input frame"}
    else:
        issue_img = scan_image(pd.Timestamp(ftime))
        issue_img["source"] = "NOAA scan at issue time"

    return jsonify(forecast_time=ftime, latest=latest,
                   times=sorted(times, reverse=True),
                   has_forecast_json=fc is not None,
                   issue_image=issue_img, columns=columns)


# ── backtest view ─────────────────────────────────────────────────────────────

def _scores(g):
    """MAE for v3 and smart persistence, skill, bias and 90% coverage."""
    g = g.dropna(subset=["sp_abs_error_wm2"])
    if g.empty:
        return None
    mae, sp = g["abs_error_wm2"].mean(), g["sp_abs_error_wm2"].mean()
    return {"n": int(len(g)),
            "mae": round(float(mae), 1),
            "sp_mae": round(float(sp), 1),
            "skill": round(float(100 * (1 - mae / sp)), 1) if sp > 0 else None,
            "bias": round(float((g["forecast_wm2"] - g["actual_wm2"]).mean()), 1),
            "picp": round(float(100 * g["in_ci"].mean()), 0)}


@app.get("/api/backtest")
def backtest():
    """Summary of verification_log_backtest.csv: v3 rows only, scored hours only."""
    if not BACKTEST_LOG.exists():
        return jsonify(error="no backtest yet"), 404
    df = pd.read_csv(BACKTEST_LOG)
    df = df[(df["source"] == "measured")
            & (df.get("method", "v3_ensemble") == "v3_ensemble")].copy()
    if df.empty:
        return jsonify(error="no backtest yet"), 404
    for c in ["forecast_wm2", "actual_wm2", "abs_error_wm2", "sp_abs_error_wm2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["in_ci"] = df["in_90ci"].astype(str).str.lower().eq("true")
    df["issue"] = pd.to_datetime(df["forecast_time"])

    by = lambda key, fmt: [dict(key=fmt(k), **s) for k, g in df.groupby(key)
                           if (s := _scores(g))]
    return jsonify(
        start=df["issue"].min().strftime("%Y-%m-%d"),
        end=df["issue"].max().strftime("%Y-%m-%d"),
        forecasts=int(df["forecast_time"].nunique()),
        days=int(df["issue"].dt.date.nunique()),
        overall=_scores(df),
        horizons=by("horizon", str),
        months=by(df["issue"].dt.to_period("M"), lambda p: p.strftime("%B %Y")),
        hours=by(df["issue"].dt.hour, lambda h: f"{h:02d}:00"),
        daily=by(df["issue"].dt.date, str))


_csv_rows = None


def csv_row(t_sgt):
    """The CSV row at an SGT hour (scan path, GHI, clear-sky), or {}."""
    global _csv_rows
    if _csv_rows is None:
        df = pd.read_csv(CSV, usecols=["timestamp", "image_path", "ghi", "ghi_clearsky"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _csv_rows = df.set_index("timestamp").to_dict("index")
    return _csv_rows.get(pd.Timestamp(t_sgt), {})


def archive_image(t_sgt):
    """Image record for the archived scan the CSV row at t_sgt points at."""
    return archive_file_image(csv_row(t_sgt).get("image_path"))


def archive_file_image(rel):
    """Image record for a data/satellite_aws/... path; the scan time is in its name."""
    if not isinstance(rel, str) or not (ROOT / rel).exists():
        return {"status": "missing"}
    stamp = Path(rel).stem.rsplit("_", 2)[-2:]          # himawari_sg_YYYYmmdd_HHMMSS
    utc = pd.Timestamp(" ".join(stamp)).strftime("%Y-%m-%d %H:%M")
    return {"status": "ready", "scan_utc": utc,
            "url": "/img/archive/" + str(Path(rel).relative_to("data/satellite_aws"))}


@app.get("/img/archive/<path:rel>")
def archive_file(rel):
    p = (ARCHIVE / rel).resolve()
    if not p.is_relative_to(ARCHIVE.resolve()) or p.suffix != ".png" or not p.exists():
        abort(404)
    return send_file(p, max_age=86400)


@app.get("/api/backtest/runs")
def backtest_runs():
    """Every backtested issue time, newest first."""
    times = sorted((pd.Timestamp(p.stem.replace("_", " ")).strftime("%Y-%m-%d %H:%M")
                    for p in BACKTEST_FC.glob("*.json")), reverse=True)
    return jsonify(times=times)


@app.get("/api/backtest/run")
def backtest_run():
    """One backtest forecast: its numbers, its log rows, its input and target scans."""
    ftime = request.args.get("forecast_time", "")
    try:
        p = BACKTEST_FC / f"{run_id_for(ftime)}.json"
    except ValueError:
        abort(400)
    if not p.exists():
        return jsonify(error=f"no backtest forecast for {ftime}"), 404
    fc = json.loads(p.read_text())
    log = pd.read_csv(BACKTEST_LOG) if BACKTEST_LOG.exists() else pd.DataFrame()
    rows = log[log["forecast_time"].astype(str) == ftime] if len(log) else log
    # Target hours are in the past by construction, so "now" is irrelevant here.
    columns = _columns(ftime, fc, rows, archive_image, pd.Timestamp.max)
    frames = (fc.get("diagnostics", {}).get("satellite_frames") or [None] * 3)
    inputs = [dict(label=lbl, **archive_file_image(f))
              for lbl, f in zip(["t (issue)", "t − 10 min", "t − 20 min"], frames)][::-1]
    return jsonify(forecast_time=ftime, diagnostics=fc.get("diagnostics", {}),
                   inputs=inputs, columns=columns)


@app.get("/api/backtest/day")
def backtest_day():
    """
    One backtest day: the scan and actual GHI at every daylight hour, and every
    forecast issued that day with its three horizons scored.
    """
    try:
        day = pd.Timestamp(request.args.get("date", "")).normalize()
    except ValueError:
        abort(400)
    log = pd.read_csv(BACKTEST_LOG) if BACKTEST_LOG.exists() else pd.DataFrame()
    if log.empty:
        return jsonify(error="no backtest yet"), 404
    log = log[pd.to_datetime(log["forecast_time"]).dt.normalize() == day]
    if log.empty:
        return jsonify(error=f"no backtest forecasts on {day:%Y-%m-%d}"), 404

    hours = []
    for h in range(8, 18):
        t = day + pd.Timedelta(hours=h)
        row = csv_row(t)
        hours.append({"time": t.strftime("%H:%M"),
                      "actual": _clean(row.get("ghi")),
                      "clearsky": _clean(round(row["ghi_clearsky"], 1)) if row else None,
                      "image": archive_image(t)})

    runs = []
    for ftime, g in log.groupby("forecast_time"):
        fc_path = BACKTEST_FC / f"{run_id_for(ftime)}.json"
        fc = json.loads(fc_path.read_text()) if fc_path.exists() else {}
        cis = {f["horizon"]: f for f in fc.get("forecasts", [])}
        hz = {}
        for _, r in g.iterrows():
            f = cis.get(r["horizon"], {})
            hz[r["horizon"]] = {"target": str(r["target_time"])[11:16],
                                "forecast": _clean(r["forecast_wm2"]),
                                "actual": _clean(r["actual_wm2"]),
                                "abs_error": _clean(r["abs_error_wm2"]),
                                "in_ci": str(r["in_90ci"]).lower() == "true",
                                "lower": f.get("ghi_lower_90"),
                                "upper": f.get("ghi_upper_90"),
                                "sp_abs_error": _clean(r["sp_abs_error_wm2"])}
        v3 = [hz[h]["abs_error"] for h in HORIZONS if h in hz]
        sp = [hz[h]["sp_abs_error"] for h in HORIZONS if h in hz
              and hz[h]["sp_abs_error"] is not None]
        runs.append({"forecast_time": ftime, "horizons": hz,
                     "mae": round(sum(v3) / len(v3), 1) if v3 else None,
                     "sp_mae": round(sum(sp) / len(sp), 1) if sp else None})
    return jsonify(date=f"{day:%Y-%m-%d}", hours=hours, runs=runs)


@app.get("/")
def index():
    return render_template("index.html", static_site=False)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8050)),
            debug=False, threaded=True)
