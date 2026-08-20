import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
from PIL import Image
from io import BytesIO
import pytz
import numpy as np


def _load_env(path=".env"):
    """Minimal .env loader (no external dependency)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()


class WeatherCollector:
    def __init__(self, save_dir="datanow/weather"):
        self.base_url = "https://api.data.gov.sg/v1/environment"
        self.endpoints = {
            "air-temperature": "air_temperature",
            "rainfall": "rainfall",
            "relative-humidity": "relative_humidity",
            "wind-direction": "wind_direction",
            "wind-speed": "wind_speed"
        }
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def fetch_data(self, date_time=None):
        """
        Fetch weather data from all endpoints. 
        date_time: datetime string in YYYY-MM-DD[T]HH:mm:ss (optional, defaults to now)
        """
        params = {}
        if date_time:
            params['date_time'] = date_time
        
        collected_data = {}
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        for name, data_key in self.endpoints.items():
            url = f"{self.base_url}/{name}"
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Check if 'items' exists and is not empty
                if 'items' in data and len(data['items']) > 0:
                    readings = data['items'][0]
                    collected_data[name] = readings
                    print(f"[{timestamp}] Fetched {name}")
                else:
                    print(f"[{timestamp}] No data for {name}")
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        # Save raw JSON
        filename = f"{self.save_dir}/weather_current.json"
        with open(filename, 'w') as f:
            json.dump(collected_data, f, indent=4)
        
        return collected_data, filename


class SatelliteCollector:
    def __init__(self, save_dir="datanow/satellite"):
        self.ftp_host = "ftp.ptree.jaxa.jp"
        self.ftp_user = os.environ.get("JAXA_FTP_USER")
        self.ftp_pass = os.environ.get("JAXA_FTP_PASS")
        if not self.ftp_user or not self.ftp_pass:
            print("⚠️  JAXA_FTP_USER / JAXA_FTP_PASS not set — add them to .env "
                  "(see .env.example). Satellite fetch will fail without them.")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def get_latest_timestamp(self):
        # Himawari updates every 10 min, ~20-30 min delay
        now    = datetime.now(pytz.UTC)
        target = now - timedelta(minutes=30)
        minute = target.minute - (target.minute % 10)
        target = target.replace(minute=minute, second=0, microsecond=0)
        return target

    # ── Image quality gate ────────────────────────────────────────────────
    # Measured from the 9,186 training images: mean brightness tracks the sun
    # (06:00 SGT ≈ 19, 11:00 ≈ 88, 17:00 ≈ 19) and NEVER drops near zero
    # during daylight. A near-black or flat tile in daylight is a bad fetch,
    # not weather — feeding it to the model silently corrupts the forecast.
    MIN_DAYLIGHT_MEAN = 3.0    # below this in daylight = dead tile
    MIN_DAYLIGHT_STD  = 2.0    # flat tile = no cloud structure

    @staticmethod
    def validate_tile(img, dt_utc):
        """Returns (ok: bool, reason: str). Night tiles are legitimately dark."""
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        if arr.ndim != 2 or min(arr.shape) < 100:
            return False, f"bad dimensions {arr.shape}"
        mean, std = float(arr.mean()), float(arr.std())
        sgt_hour = (dt_utc + timedelta(hours=8)).hour
        if 7 <= sgt_hour <= 18:      # daylight — brightness is expected
            if mean < SatelliteCollector.MIN_DAYLIGHT_MEAN:
                return False, f"black tile in daylight (mean={mean:.2f})"
            if std < SatelliteCollector.MIN_DAYLIGHT_STD:
                return False, f"flat/uniform tile (std={std:.2f})"
        return True, f"ok (mean={mean:.1f}, std={std:.1f})"

    def fetch_image(self, date_time=None, out_name="himawari_current.png"):
        """
        Fetch a validated Singapore tile, trying each configured source in
        order until one returns an image that passes validate_tile().

        Source order is set by SATELLITE_SOURCES in .env (comma-separated).
        Default: "nict,jaxa".
        """
        sources = [s.strip() for s in
                   os.environ.get("SATELLITE_SOURCES", "nict,jaxa").split(",")
                   if s.strip()]
        for src in sources:
            fn = {"nict": self.fetch_image_nict,
                  "slider": self.fetch_image_slider,
                  "gk2a": self.fetch_image_gk2a,
                  "jaxa": self.fetch_image_jaxa}.get(src)
            if fn is None:
                print(f"  ⚠️  Unknown satellite source '{src}' — skipping")
                continue
            try:
                result = fn(date_time, out_name=out_name)
            except Exception as e:
                print(f"  ✗ Source '{src}' raised: {e}")
                result = None
            if result:
                return result
            print(f"  Source '{src}' failed — trying next...")

        # Last resort: reuse the most recent good image rather than a black one
        stale = f"{self.save_dir}/{out_name}"
        if os.path.exists(stale):
            age_h = (time.time() - os.path.getmtime(stale)) / 3600
            print(f"  ⚠️  ALL SOURCES FAILED — keeping previous image "
                  f"({age_h:.1f}h old). Forecast will be degraded.")
            return stale
        print("  ✗ All sources failed and no previous image available")
        return None

    def fetch_frame_series(self, date_time=None):
        """
        Fetch the 3 hourly frames the v2 model expects: t, t-1h, t-2h.

        Training fed the model three CONSECUTIVE hourly frames (and computed
        optical flow between them). Fetching only the current frame and
        zero-filling t-1/t-2 is a train/inference mismatch that destroys the
        flow signal, so always fetch the series for inference.

        Returns (current_path, prev1_path, prev2_path); entries may be None.
        """
        if date_time is None:
            date_time = self.get_latest_timestamp()
        paths = []
        for h, name in [(0, "himawari_current.png"),
                        (1, "himawari_prev1.png"),
                        (2, "himawari_prev2.png")]:
            paths.append(
                self.fetch_image(date_time - timedelta(hours=h),
                                 out_name=name))
        got = sum(p is not None for p in paths)
        print(f"  Frame series: {got}/3 frames fetched (t, t-1h, t-2h)")
        return tuple(paths)

    def _save_tile(self, img, out_name, source, dt_utc):
        """Grayscale→RGB (matching training preprocessing) and save."""
        gray = img.convert("L")
        img_rgb = Image.merge("RGB", (gray, gray, gray))
        filename = f"{self.save_dir}/{out_name}"
        img_rgb.save(filename)
        print(f"  ✓ [{source}] saved {out_name} "
              f"({img_rgb.size[0]}x{img_rgb.size[1]}, "
              f"{dt_utc.strftime('%H:%M')} UTC)")
        return filename

    def fetch_image_nict(self, date_time=None, out_name="himawari_current.png",
                         max_attempts=7):
        """
        Himawari tile from NICT — the SAME source and preprocessing as the
        training images, so the model sees the distribution it learned on.

        NICT intermittently publishes dead/black tiles. Rather than accepting
        one, step back in 10-minute increments (NICT's native cadence) until a
        tile passes validation, up to ~1 hour.
        """
        if date_time is None:
            date_time = self.get_latest_timestamp()
        date_time = (pytz.UTC.localize(date_time) if date_time.tzinfo is None
                     else date_time.astimezone(pytz.UTC))

        print(f"  [nict] target {date_time.strftime('%Y-%m-%d %H:%M')} UTC")
        for attempt in range(max_attempts):
            t = date_time - timedelta(minutes=10 * attempt)
            # Same tile as himawari_data.py: level 4d, tile (1,1) = Singapore
            url = (f"https://himawari8-dl.nict.go.jp/himawari8/img/D531106"
                   f"/4d/550/{t.strftime('%Y/%m/%d/%H%M00')}_1_1.png")
            try:
                resp = requests.get(url, verify=False, timeout=30)
                if resp.status_code != 200:
                    print(f"    · {t.strftime('%H:%M')} → HTTP {resp.status_code}")
                    continue
                img = Image.open(BytesIO(resp.content))
                ok, reason = self.validate_tile(img, t)
                if not ok:
                    print(f"    · {t.strftime('%H:%M')} → rejected: {reason}")
                    continue
                if attempt:
                    print(f"    (used {attempt*10} min older frame)")
                return self._save_tile(img, out_name, "nict", t)
            except Exception as e:
                print(f"    · {t.strftime('%H:%M')} → error: {str(e)[:60]}")
        print(f"  ✗ [nict] no valid tile in the last "
              f"{max_attempts*10} minutes")
        return None

    # ── Alternative sources ───────────────────────────────────────────────
    # ⚠️  UNVERIFIED: the two providers below are wired up but have NOT been
    # tested against a live endpoint. Enable one by setting, in .env:
    #     SATELLITE_SOURCES=nict,slider,jaxa
    # and check the printed output. If a URL pattern has changed, fix it here.
    #
    # ⚠️  IMPORTANT — GK-2A is a DIFFERENT SATELLITE (KMA, 128.2°E), not
    # Himawari. The model was trained exclusively on Himawari NICT tiles, so
    # GK-2A imagery is out-of-distribution: different sensor, resolution,
    # viewing geometry and calibration. Using it WITHOUT retraining the CNN on
    # GK-2A images will likely make forecasts worse, not better. Prefer
    # 'slider' (a different provider of the SAME Himawari data) if you just
    # want redundancy.

    def fetch_image_slider(self, date_time=None,
                           out_name="himawari_current.png", max_attempts=4):
        """RAMMB/CIRA SLIDER — different PROVIDER, same Himawari sensor."""
        if date_time is None:
            date_time = self.get_latest_timestamp()
        date_time = (pytz.UTC.localize(date_time) if date_time.tzinfo is None
                     else date_time.astimezone(pytz.UTC))
        base = "https://rammb-slider.cira.colostate.edu/data/imagery"
        for attempt in range(max_attempts):
            t = date_time - timedelta(minutes=10 * attempt)
            url = (f"{base}/{t.strftime('%Y%m%d')}/himawari---full_disk/"
                   f"geocolor/{t.strftime('%Y%m%d%H%M%S')}/02/003_002.png")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"    · [slider] {t.strftime('%H:%M')} → "
                          f"HTTP {resp.status_code}")
                    continue
                img = Image.open(BytesIO(resp.content))
                ok, reason = self.validate_tile(img, t)
                if not ok:
                    print(f"    · [slider] rejected: {reason}")
                    continue
                return self._save_tile(img, out_name, "slider", t)
            except Exception as e:
                print(f"    · [slider] error: {str(e)[:60]}")
        return None

    def fetch_image_gk2a(self, date_time=None,
                         out_name="himawari_current.png", max_attempts=3):
        """
        GK-2A (KMA, 128.2°E) via NMSC public imagery — a genuinely different
        satellite. See the domain-shift warning above: retrain before relying
        on this for forecasts.
        """
        if date_time is None:
            date_time = self.get_latest_timestamp()
        date_time = (pytz.UTC.localize(date_time) if date_time.tzinfo is None
                     else date_time.astimezone(pytz.UTC))
        base = "https://nmsc.kma.go.kr/IMG/GK2A/AMI/PRIMARY/L1B/COMPLETE/EA"
        for attempt in range(max_attempts):
            t = date_time - timedelta(minutes=10 * attempt)
            url = (f"{base}/{t.strftime('%Y%m/%d/%H')}/"
                   f"gk2a_ami_le1b_vi006_ea020lc_{t.strftime('%Y%m%d%H%M')}.srv.png")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"    · [gk2a] {t.strftime('%H:%M')} → "
                          f"HTTP {resp.status_code}")
                    continue
                img = Image.open(BytesIO(resp.content))
                ok, reason = self.validate_tile(img, t)
                if not ok:
                    print(f"    · [gk2a] rejected: {reason}")
                    continue
                print("  ⚠️  GK-2A is a different satellite than the model was "
                      "trained on — expect degraded accuracy")
                return self._save_tile(img, out_name, "gk2a", t)
            except Exception as e:
                print(f"    · [gk2a] error: {str(e)[:60]}")
        return None

    def fetch_image_jaxa(self, date_time=None,
                         out_name="himawari_current.png"):
        import ftplib
        import netCDF4 as nc

        if not self.ftp_user or not self.ftp_pass:
            print("  ✗ No JAXA credentials in .env — skipping FTP fallback")
            return None

        if date_time is None:
            date_time = self.get_latest_timestamp()

        if date_time.tzinfo is None:
            date_time = pytz.UTC.localize(date_time)
        else:
            date_time = date_time.astimezone(pytz.UTC)

        yyyymm = date_time.strftime("%Y%m")
        dd     = date_time.strftime("%d")
        hhmm   = date_time.strftime("%H%M")

        # Singapore bounds with padding
        sg_lat, sg_lon = 1.3521, 103.8198
        pad = 5.0

        print(f"Fetching Satellite Image via JAXA FTP")
        print(f"  Time (UTC): {date_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            ftp = ftplib.FTP(self.ftp_host, timeout=30)
            ftp.login(self.ftp_user, self.ftp_pass)
            ftp.cwd(f"/jma/netcdf/{yyyymm}/{dd}")
            files = ftp.nlst()

            # Find R21 full-disk file closest to target time
            r21 = [f for f in files if "R21" in f and hhmm in f and "FLDK" in f]

            # If exact time not found, find closest R21 file
            if not r21:
                r21_all = [f for f in files if "R21" in f and "FLDK" in f]
                if not r21_all:
                    print("  ⚠️  No R21 files found — trying adjacent times")
                    ftp.quit()
                    return None

                # Pick closest by time (proper datetime diff, handles
                # midnight wraparound e.g. 2350 vs 0000)
                def file_time_diff(f):
                    try:
                        parts = f.split("_")
                        file_dt = datetime.strptime(parts[2] + parts[3],
                                                    "%Y%m%d%H%M")
                        file_dt = pytz.UTC.localize(file_dt)
                        return abs((file_dt - date_time).total_seconds())
                    except (ValueError, IndexError):
                        return float("inf")

                r21 = [sorted(r21_all, key=file_time_diff)[0]]

            target_file = r21[0]
            print(f"  Downloading: {target_file}")

            buf = BytesIO()
            ftp.retrbinary(f"RETR {target_file}", buf.write)
            ftp.quit()

            # Parse NetCDF
            buf.seek(0)
            ds  = nc.Dataset("dummy", memory=buf.read())
            lat = ds.variables["latitude"][:]
            lon = ds.variables["longitude"][:]
            alb = ds.variables["albedo_03"][:]

            # Extract Singapore tile
            lat_idx = np.where((lat >= sg_lat - pad) & (lat <= sg_lat + pad))[0]
            lon_idx = np.where((lon >= sg_lon - pad) & (lon <= sg_lon + pad))[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                print("  ⚠️  Singapore not in coverage area")
                return None

            tile = alb[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

            # Handle masked/fill values
            if hasattr(tile, 'filled'):
                tile = tile.filled(0.0)
            tile = np.clip(tile.astype(np.float32), 0, 1)

            # Convert to RGB (replicate across 3 channels)
            tile_rgb = np.stack([tile, tile, tile], axis=-1)  # (H, W, 3)

            # Save as PNG
            img = Image.fromarray((tile_rgb * 255).astype(np.uint8))
            filename = f"{self.save_dir}/{out_name}"
            img.save(filename)
            print(f"✓ Satellite image saved to {filename} (shape: {tile.shape})")
            return filename

        except Exception as e:
            print(f"✗ Error fetching via FTP: {e}")
            return None


class CurrentDataCollector:
    """
    Combines weather and satellite data collection for current conditions.
    """
    def __init__(self, weather_dir="datanow/weather", satellite_dir="datanow/satellite"):
        self.weather_collector = WeatherCollector(save_dir=weather_dir)
        self.satellite_collector = SatelliteCollector(save_dir=satellite_dir)
        self.combined_dir = "datanow/current"
        os.makedirs(self.combined_dir, exist_ok=True)
    
    def fetch_current_data(self, date_time=None):
        """
        Fetch both weather data and the corresponding satellite image.

        Args:
            date_time: optional ISO string "YYYY-MM-DDTHH:MM:SS" (SGT).
                       If given, fetches data for that time instead of now.

        Returns a dictionary with paths to both data files.
        """
        print("\n" + "="*70)
        print("FETCHING CURRENT DATA")
        print("="*70 + "\n")

        timestamp = datetime.now()

        # Parse target time for the satellite fetch (needs a datetime, UTC-aware)
        sat_datetime = None
        if date_time:
            sgt = pytz.timezone("Asia/Singapore")
            sat_datetime = sgt.localize(datetime.fromisoformat(date_time))

        # 1. Fetch weather data
        print("📊 Fetching Weather Data...")
        print("-" * 70)
        weather_data, weather_file = self.weather_collector.fetch_data(
            date_time=date_time)

        print("\n")

        # 2. Fetch satellite image
        print("🛰️  Fetching Satellite Image...")
        print("-" * 70)
        # Fetch t, t-1h, t-2h (v2 model needs the multi-frame stack)
        frames = self.satellite_collector.fetch_frame_series(
            date_time=sat_datetime)
        satellite_file = frames[0]
        if satellite_file is None:   # NICT series failed → JAXA single-frame
            satellite_file = self.satellite_collector.fetch_image_jaxa(
                date_time=sat_datetime)
        
        # 3. Create combined metadata
        metadata = {
            "collection_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "weather_data_file": weather_file,
            "satellite_image_file": satellite_file,
            "weather_summary": self._summarize_weather(weather_data)
        }
        
        # Save metadata
        metadata_file = f"{self.combined_dir}/current_summary.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("\n" + "="*70)
        print("✓ DATA COLLECTION COMPLETE")
        print("="*70)
        print(f"\nFiles saved:")
        print(f"  Weather:   {weather_file}")
        print(f"  Satellite: {satellite_file}")
        print(f"  Metadata:  {metadata_file}")
        
        if metadata['weather_summary']:
            print(f"\n📈 Weather Summary:")
            for key, value in metadata['weather_summary'].items():
                print(f"  {key}: {value}")
        
        print("\n")
        
        return metadata
    
    def _summarize_weather(self, weather_data):
        """
        Extract key weather metrics from the collected data.
        """
        summary = {}
        
        try:
            # Temperature
            if 'air-temperature' in weather_data:
                readings = weather_data['air-temperature'].get('readings', [])
                if readings:
                    temps = [r['value'] for r in readings if 'value' in r]
                    if temps:
                        summary['avg_temperature_c'] = round(sum(temps) / len(temps), 1)
            
            # Humidity
            if 'relative-humidity' in weather_data:
                readings = weather_data['relative-humidity'].get('readings', [])
                if readings:
                    humidity = [r['value'] for r in readings if 'value' in r]
                    if humidity:
                        summary['avg_humidity_pct'] = round(sum(humidity) / len(humidity), 1)
            
            # Rainfall
            if 'rainfall' in weather_data:
                readings = weather_data['rainfall'].get('readings', [])
                if readings:
                    rainfall = [r['value'] for r in readings if 'value' in r]
                    if rainfall:
                        summary['avg_rainfall_mm'] = round(sum(rainfall) / len(rainfall), 2)
            
            # Wind Speed
            if 'wind-speed' in weather_data:
                readings = weather_data['wind-speed'].get('readings', [])
                if readings:
                    wind_speed = [r['value'] for r in readings if 'value' in r]
                    if wind_speed:
                        summary['avg_wind_speed_kmh'] = round(sum(wind_speed) / len(wind_speed), 1)
            
        except Exception as e:
            print(f"Warning: Error creating weather summary: {e}")
        
        return summary
    
    def start_polling(self, interval_minutes=10):
        """
        Continuously collect data at specified intervals.
        
        Args:
            interval_minutes: How often to collect data (default: 10 minutes)
        """
        interval_seconds = interval_minutes * 60
        print(f"\n🔄 Starting continuous data collection every {interval_minutes} minutes...")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.fetch_current_data()
                print(f"\n⏳ Waiting {interval_minutes} minutes until next collection...")
                print(f"   Next collection at: {(datetime.now() + timedelta(seconds=interval_seconds)).strftime('%H:%M:%S')}\n")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\n⏹️  Polling stopped by user.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str, default=None, 
                        help="Target time for verification (ISO format: YYYY-MM-DDTHH:MM:SS)")
    args = parser.parse_args()
    
    collector = CurrentDataCollector()
    
    if args.time:
        target_dt_sgt = pytz.timezone("Asia/Singapore").localize(
            datetime.fromisoformat(args.time))
        print(f"Fetching weather and satellite data for "
              f"{target_dt_sgt.strftime('%Y-%m-%d %H:%M:%S SGT')}...\n")
        collector.fetch_current_data(date_time=args.time)
    else:
        print("Fetching current weather and satellite data...\n")
        collector.fetch_current_data()
