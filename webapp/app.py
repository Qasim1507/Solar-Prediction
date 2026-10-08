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
"""
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

sys.path.insert(0, str(ROOT))
app = Flask(__name__)


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


def _run(job):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.Popen([str(PY), "-u", *COMMANDS[job.kind]], cwd=ROOT,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env,
                                bufsize=1)
        for line in proc.stdout:
            job.lines.append(line.rstrip("\n"))
        job.returncode = proc.wait()
    except Exception as e:
        job.lines.append(f"[webapp] failed to start: {e}")
        job.returncode = -1
    if job.returncode == 0 and job.kind == "predict":
        try:
            run_id = archive_run()
            job.lines.append(f"[webapp] archived run -> webapp/runs/{run_id}/")
        except Exception as e:
            job.lines.append(f"[webapp] could not archive run: {e}")
    job.state = "done" if job.returncode == 0 else "failed"


@app.post("/api/<kind>")
def start_job(kind):
    global _job
    if kind not in COMMANDS:
        abort(404)
    with _job_lock:
        if _job is not None and _job.state == "running":
            return jsonify(error=f"a {_job.kind} job is already running"), 409
        _job = Job(kind)
        threading.Thread(target=_run, args=(_job,), daemon=True).start()
    return jsonify(kind=kind, started=_job.started)


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
        out.append({"name": name,
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
    rows = log[log["forecast_time"].astype(str) == ftime] if len(log) else log
    by_h = {r["horizon"]: r for _, r in rows.iterrows()}
    fc_by_h = {f["horizon"]: f for f in (fc or {}).get("forecasts", [])}
    now = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None)

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
        col["image"] = (scan_image(t) if t <= now
                        else {"status": "pending"})
        columns.append(col)

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


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8050)),
            debug=False, threaded=True)
