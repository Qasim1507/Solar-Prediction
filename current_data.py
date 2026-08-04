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

    def fetch_image(self, date_time=None):
        """
        Fetch the latest Singapore satellite tile.
        Primary: NICT public Himawari tile server (no login) — the SAME
                 source as the training images (data/satellite/).
        Fallback: JAXA P-Tree FTP (needs JAXA_FTP_USER/PASS in .env).
        """
        result = self.fetch_image_nict(date_time)
        if result:
            return result
        print("  NICT fetch failed — trying JAXA FTP fallback...")
        return self.fetch_image_jaxa(date_time)

    def fetch_image_nict(self, date_time=None):
        """Fetch Himawari tile from NICT (public, matches training data)."""
        if date_time is None:
            date_time = self.get_latest_timestamp()
        if date_time.tzinfo is None:
            date_time = pytz.UTC.localize(date_time)
        else:
            date_time = date_time.astimezone(pytz.UTC)

        # Same tile as himawari_data.py: level 4d, row 1, col 1 (Singapore)
        date_str = date_time.strftime("%Y/%m/%d/%H%M00")
        url = (f"https://himawari8-dl.nict.go.jp/himawari8/img/D531106"
               f"/4d/550/{date_str}_1_1.png")
        print(f"Fetching Satellite Image via NICT (public)")
        print(f"  Time (UTC): {date_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            resp = requests.get(url, verify=False, timeout=30)
            if resp.status_code != 200:
                print(f"  ✗ NICT returned status {resp.status_code}")
                return None
            img = Image.open(BytesIO(resp.content))
            # Grayscale → RGB, matching training preprocessing exactly
            gray = img.convert("L")
            img_rgb = Image.merge("RGB", (gray, gray, gray))
            filename = f"{self.save_dir}/himawari_current.png"
            img_rgb.save(filename)
            print(f"✓ Satellite image saved to {filename} "
                  f"(size: {img_rgb.size}, grayscale-RGB)")
            return filename
        except Exception as e:
            print(f"  ✗ NICT error: {e}")
            return None

    def fetch_image_jaxa(self, date_time=None):
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
            filename = f"{self.save_dir}/himawari_current.png"
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
        satellite_file = self.satellite_collector.fetch_image(
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
