import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
from PIL import Image
from io import BytesIO
import pytz

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
        self.base_url = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/1d/550"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def get_latest_timestamp(self):
        # Himawari-8 updates every 10 minutes, usually with a ~20-30 min delay
        now = datetime.now(pytz.UTC)
        delta = timedelta(minutes=30)
        target = now - delta
        # Round down to nearest 10 minutes
        minute = target.minute - (target.minute % 10)
        target = target.replace(minute=minute, second=0, microsecond=0)
        return target
    
    def fetch_image(self, date_time=None):
        """
        Fetch a specific tile (Row 1, Col 1) from the 4d (4x4 grid) level.
        Singapore (1.35N, 103.8E) is typically in this tile for Himawari-8 (140.7E).
        """
        if date_time is None:
            date_time = self.get_latest_timestamp()
        
        # Ensure datetime is in UTC for API request
        if date_time.tzinfo is None:
            date_time = pytz.UTC.localize(date_time)
        else:
            date_time = date_time.astimezone(pytz.UTC)
        
        # Format: YYYY/MM/DD/HHmm00_R_C.png
        level = "4d"
        tile_r = 1
        tile_c = 1
        
        date_str = date_time.strftime("%Y/%m/%d/%H%M00")
        base = self.base_url.replace("1d", level) 
        url = f"{base}/{date_str}_{tile_r}_{tile_c}.png"
        
        print(f"Fetching Satellite Tile (Singapore Focus)")
        print(f"  Time (UTC): {date_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url, verify=False, timeout=30)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Save
                filename = f"{self.save_dir}/himawari_current.png"
                img.save(filename)
                print(f"✓ Satellite image saved to {filename}")
                return filename
            else:
                print(f"✗ Failed to fetch image. Status: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Error fetching satellite image: {e}")
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
    
    def fetch_current_data(self):
        """
        Fetch both current weather data and corresponding satellite image.
        Returns a dictionary with paths to both data files.
        """
        print("\n" + "="*70)
        print("FETCHING CURRENT DATA")
        print("="*70 + "\n")
        
        timestamp = datetime.now()
        
        # 1. Fetch weather data
        print("📊 Fetching Weather Data...")
        print("-" * 70)
        weather_data, weather_file = self.weather_collector.fetch_data()
        
        print("\n")
        
        # 2. Fetch satellite image
        print("🛰️  Fetching Satellite Image...")
        print("-" * 70)
        satellite_file = self.satellite_collector.fetch_image()
        
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
    collector = CurrentDataCollector()
    
    print("Fetching current weather and satellite data...\n")
    collector.fetch_current_data()
