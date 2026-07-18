"""
fetch_pv_ground_truth.py

Fetches the best available free PV power ground truth data for Singapore,
combining two sources to cover your full 2020–2026 date range:

  2005–2023 → PVGIS 5.3 (EU JRC, ERA5 database, free, no key)
              Satellite-informed reanalysis. Better than pvlib.

  2024–2026 → Open-Meteo Historical + pvlib simulation
              ERA5 reanalysis weather → pvlib PV model.
              Same source as your existing pv_power_predicted column,
              but recalculated cleanly here for consistency.

Output:
  data/pv_ground_truth_sg.csv
  Columns: timestamp (SGT naive), pv_actual (W)

Usage:
  pip install requests pandas pvlib pytz
  python fetch_pv_ground_truth.py
"""

import requests
import pandas as pd
import numpy as np
import pvlib
import pytz
import os
import time

# ---------------------------------------------------------------------------
# System config
# ---------------------------------------------------------------------------
LAT         = 1.3521
LON         = 103.8198
PEAK_POWER  = 1       # kWp — normalised; scale up if you know your system size
TILT        = 10      # degrees from horizontal
AZIMUTH     = 0       # PVGIS: 0=south. For Singapore (equatorial), 0 is fine.
SYSTEM_LOSS = 14      # % losses
PV_TECH     = "crystSi"

OUTPUT_PATH = "data/pv_ground_truth_sg.csv"
SGT         = pytz.timezone("Asia/Singapore")


# ---------------------------------------------------------------------------
# Part 1: PVGIS v5.3 — covers 2005–2023 via ERA5
# ---------------------------------------------------------------------------

def fetch_pvgis_year(year: int) -> pd.DataFrame | None:
    """Fetch one year from PVGIS 5.3 API (ERA5 database, global coverage)."""
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat":           LAT,
        "lon":           LON,
        "startyear":     year,
        "endyear":       year,
        "pvcalculation": 1,
        "peakpower":     PEAK_POWER,
        "loss":          SYSTEM_LOSS,
        "angle":         TILT,
        "aspect":        AZIMUTH,
        "pvtechchoice":  PV_TECH,
        "raddatabase":   "PVGIS-ERA5",   # explicit — only global DB for SG
        "outputformat":  "json",
        "browser":       0,
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        hourly = resp.json()["outputs"]["hourly"]
        df = pd.DataFrame(hourly)
        # PVGIS timestamp format: "20200101:0000"
        df["timestamp"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True)
        df = df.rename(columns={"P": "pv_actual"})[["timestamp", "pv_actual"]]
        print(f"  ✓ PVGIS {year}: {len(df):,} rows | "
              f"range {df['pv_actual'].min():.0f}–{df['pv_actual'].max():.0f} W")
        return df
    except requests.HTTPError as e:
        print(f"  ✗ PVGIS {year}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ PVGIS {year}: {e}")
        return None


def fetch_pvgis_range(start_year: int, end_year: int) -> pd.DataFrame:
    print(f"\n[PVGIS 5.3] Fetching {start_year}–{end_year} ...")
    dfs = []
    years = list(range(start_year, end_year + 1))
    for i, year in enumerate(years, 1):
        print(f"  [{i}/{len(years)}] Year {year}")
        df = fetch_pvgis_year(year)
        if df is not None:
            dfs.append(df)
        if i < len(years):
            time.sleep(1.2)

    if not dfs:
        print("  No PVGIS data fetched.")
        return pd.DataFrame()

    combined = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    # Convert UTC → SGT naive
    combined["timestamp"] = (
        combined["timestamp"]
        .dt.tz_convert(SGT)
        .dt.tz_localize(None)
    )
    # ERA5 timestamps are period-centre (HH:30) — floor to hour to match dataset
    combined["timestamp"] = combined["timestamp"].dt.floor("h")
    return combined


# ---------------------------------------------------------------------------
# Part 2: Open-Meteo + pvlib — covers 2024–present
# ---------------------------------------------------------------------------

def fetch_openmeteo_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo archive (ERA5, free, no key)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": (
            "temperature_2m,"
            "shortwave_radiation,"
            "direct_normal_irradiance,"
            "diffuse_radiation"
        ),
        "timezone": "Asia/Singapore",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def simulate_pv_pvlib(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate PV output using pvlib on top of Open-Meteo weather.
    Returns DataFrame with columns: timestamp, pv_actual (W)
    """
    location = pvlib.location.Location(LAT, LON, tz="Asia/Singapore")
    times = pd.DatetimeIndex(weather_df["timestamp"], tz="Asia/Singapore")

    # Clear-sky for filtering
    clearsky = location.get_clearsky(times)

    # Temperature derating: -0.4% per °C above 25°C
    temp_coeff = -0.004
    pv = (
        weather_df["shortwave_radiation"]
        * PEAK_POWER          # W per kWp
        * (1 - SYSTEM_LOSS / 100)
        * (1 + temp_coeff * (weather_df["temperature_2m"] - 25))
    ).clip(lower=0)

    df = pd.DataFrame({
        "timestamp":   weather_df["timestamp"],
        "pv_actual":   pv.values,
        "ghi_clearsky": clearsky["ghi"].values,
    })

    # Filter to daylight only (clearsky GHI > 0)
    df = df[df["ghi_clearsky"] > 0].drop(columns=["ghi_clearsky"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.reset_index(drop=True)


def fetch_openmeteo_pv(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"\n[Open-Meteo + pvlib] Fetching {start_date} → {end_date} ...")
    try:
        weather = fetch_openmeteo_weather(start_date, end_date)
        df = simulate_pv_pvlib(weather)
        print(f"  ✓ {len(df):,} daylight rows | "
              f"range {df['pv_actual'].min():.0f}–{df['pv_actual'].max():.0f} W")
        return df
    except Exception as e:
        print(f"  ✗ Open-Meteo error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all():
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    print("\n" + "="*60)
    print("FETCHING PV GROUND TRUTH — SINGAPORE")
    print(f"  2020–2023: PVGIS 5.3 ERA5 (satellite-informed reanalysis)")
    print(f"  2024–2026: Open-Meteo ERA5 + pvlib simulation")
    print(f"  System: {PEAK_POWER} kWp, tilt={TILT}°, loss={SYSTEM_LOSS}%")
    print("="*60)

    # ---- PVGIS: 2020–2023 ----
    pvgis_df = fetch_pvgis_range(start_year=2020, end_year=2023)

    # ---- Open-Meteo + pvlib: 2024–present ----
    import datetime as _dt
    end_date = (_dt.date.today() - _dt.timedelta(days=5)).isoformat()
    openmeteo_df = fetch_openmeteo_pv("2024-01-01", end_date)

    # ---- Combine ----
    parts = [df for df in [pvgis_df, openmeteo_df] if not df.empty]
    if not parts:
        print("\n✗ No data fetched at all. Check internet connection.")
        return

    combined = (
        pd.concat(parts)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Filter to daylight hours (08:00–17:00 SGT, matching combined_dataset.csv)
    combined = combined[
        (combined["timestamp"].dt.hour >= 8) &
        (combined["timestamp"].dt.hour <= 17)
    ].reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(combined):,} rows → {OUTPUT_PATH}")
    print(f"  Full range:  {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    print(f"  pv_actual:   {combined['pv_actual'].min():.1f}–{combined['pv_actual'].max():.1f} W")
    print(f"\n  Source breakdown:")
    pvgis_rows = combined[combined["timestamp"].dt.year <= 2023]
    om_rows    = combined[combined["timestamp"].dt.year >= 2024]
    print(f"    PVGIS 5.3  (2020–2023): {len(pvgis_rows):,} rows")
    print(f"    Open-Meteo (2024–2026): {len(om_rows):,} rows")
    print(f"\n  Next step:")
    print(f"    Pass this as your target to the training pipeline:")
    print(f"    → merge on 'timestamp' with combined_dataset.csv")
    print(f"    → use 'pv_actual' as TARGET_COL instead of pv_power_predicted")
    print("="*60 + "\n")


if __name__ == "__main__":
    fetch_all()