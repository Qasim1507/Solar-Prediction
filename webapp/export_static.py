"""
webapp/export_static.py — snapshot the web app into site/ for static hosting.

    .venv/bin/python webapp/export_static.py      # then commit site/ and push

Run it after the manual predict / verify. It calls the app's own endpoints
through Flask's test client, so the static site shows exactly what the local
app shows, and writes:

    site/index.html, site/static/          the page, with STATIC_SITE = true
    site/data/forecast.json, frames.json   /api/forecast, /api/frames
    site/data/verification/<run_id>.json   /api/verification per issue time
    site/data/verification/latest.json     ... for the latest forecast
    site/data/backtest.json                /api/backtest (test-split replay summary)
    site/data/backtest/days/<date>.json    /api/backtest/day, one per day
    site/data/backtest/runs/<run_id>.json  /api/backtest/run, one per forecast
    site/img/archive/...                   their archived scans, shrunk to 256 px
    site/data/meta.json                    export time
    site/img/...                           satellite images, re-encoded as JPEG

site/ is rebuilt from scratch on every export. Target-hour scans come from the
local NOAA cache (datanow/satellite/aws_cache), so only new ones are downloaded.
"""
import io
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from flask import render_template
from PIL import Image

from app import HERE, ROOT, app, run_id_for

OUT = ROOT / "site"
SCAN_WAIT = 240  # seconds to wait for NOAA downloads per issue time


def write_json(rel, obj):
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, separators=(",", ":")))


def save_image(client, url, size=None):
    """
    Fetch an app image URL, save it as JPEG under site/, return its new URL.
    size shrinks it to size x size px. An image already saved this export is
    reused, since backtest days share most of their scans.
    """
    rel = re.sub(r"\.png$", "", url.split("?")[0].lstrip("/")) + ".jpg"
    dest = OUT / rel
    if dest.exists():
        return "/" + rel
    r = client.get(url)
    if r.status_code != 200:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(io.BytesIO(r.data)).convert("RGB")
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    im.save(dest, "JPEG", quality=82, optimize=True)
    return "/" + rel


def export_verification(client, ftime):
    url = f"/api/verification?forecast_time={quote(ftime)}"
    deadline = time.time() + SCAN_WAIT
    while True:
        d = client.get(url).get_json()
        images = [d["issue_image"], *(c["image"] for c in d["columns"])]
        if not any(i.get("status") == "fetching" for i in images):
            break
        if time.time() > deadline:
            print(f"  gave up waiting for scans for {ftime}")
            break
        time.sleep(2)
    for img in images:
        if img.get("status") == "ready":
            img["url"] = save_image(client, img["url"])
            if img["url"] is None:
                img["status"] = "missing"
        elif img.get("status") == "fetching":
            img["status"] = "missing"
    return d


BT_SIZE = 256  # px; ~16 KB per scan, ~3,500 scans for the whole test split


def _shrink(client, img):
    if img.get("status") == "ready":
        img["url"] = save_image(client, img["url"], BT_SIZE)
        if img["url"] is None:
            img["status"] = "missing"


def export_backtest_runs(client):
    """Every backtest day and forecast as JSON, with its scans at BT_SIZE px."""
    times = client.get("/api/backtest/runs").get_json()["times"]
    write_json("data/backtest/runs.json", {"times": times})
    days = sorted({t[:10] for t in times})
    for i, day in enumerate(days, 1):
        print(f"backtest day {i}/{len(days)}: {day}", end="\r")
        d = client.get(f"/api/backtest/day?date={day}").get_json()
        for h in d["hours"]:
            _shrink(client, h["image"])
        write_json(f"data/backtest/days/{day}.json", d)
    print()
    for t in times:
        d = client.get(f"/api/backtest/run?forecast_time={quote(t)}").get_json()
        for img in [*d["inputs"], *(c["image"] for c in d["columns"])]:
            _shrink(client, img)
        write_json(f"data/backtest/runs/{run_id_for(t)}.json", d)
    print(f"backtest: {len(days)} days, {len(times)} forecasts")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copytree(HERE / "static", OUT / "static")

    with app.test_request_context():
        (OUT / "index.html").write_text(
            render_template("index.html", static_site=True))

    client = app.test_client()
    write_json("data/forecast.json", client.get("/api/forecast").get_json())

    frames = client.get("/api/frames").get_json()
    for f in frames:
        if f["mtime"]:
            f["url"] = save_image(client, f["url"])
    write_json("data/frames.json", frames)

    latest = client.get("/api/verification")
    if latest.status_code == 200:
        times = latest.get_json()["times"]
        for i, ftime in enumerate(times):
            print(f"verification {i + 1}/{len(times)}: {ftime}")
            d = export_verification(client, ftime)
            write_json(f"data/verification/{run_id_for(ftime)}.json", d)
            if ftime == d["latest"]:
                write_json("data/verification/latest.json", d)

    bt = client.get("/api/backtest")
    if bt.status_code == 200:
        write_json("data/backtest.json", bt.get_json())
        export_backtest_runs(client)

    now = pd.Timestamp.now(tz="Asia/Singapore").strftime("%Y-%m-%d %H:%M")
    write_json("data/meta.json", {"exported_sgt": now})
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)}/ ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
