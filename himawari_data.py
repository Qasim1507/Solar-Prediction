import requests
import os
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import time
import pytz

class SatelliteCollector:
    def __init__(self, save_dir="data/satellite"):
        self.base_url = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/1d/550"
        self.save_dir = save_dir
        self.singapore_tz = pytz.timezone('Asia/Singapore')
        os.makedirs(save_dir, exist_ok=True)
    
    def get_latest_timestamp(self):
        # Himawari-8 updates every 10 minutes, usually with a ~20-30 min delay
        # We round down to the nearest 10-minute interval and subtract 30 minutes to be safe
        now = datetime.utcnow()
        delta = timedelta(minutes=30)
        target = now - delta
        # Round down to nearest 10 minutes
        minute = target.minute - (target.minute % 10)
        target = target.replace(minute=minute, second=0, microsecond=0)
        return target
    
    def is_daylight_singapore(self, utc_datetime):
        """
        Check if it's daylight hours in Singapore (6am - 6pm local time).
        
        Args:
            utc_datetime: datetime object in UTC
            
        Returns:
            bool: True if daylight hours, False if nighttime
        """
        # Convert UTC to Singapore time
        if utc_datetime.tzinfo is None:
            utc_datetime = pytz.UTC.localize(utc_datetime)
        
        sg_time = utc_datetime.astimezone(self.singapore_tz)
        hour = sg_time.hour
        
        # Consider 6am to 6pm as daylight hours for PV generation
        return 6 <= hour < 18
    
    def fetch_image(self, date_time=None):
        """
        Fetch a specific tile (Row 1, Col 1) from the 4d (4x4 grid) level.
        Singapore (1.35N, 103.8E) is typically in this tile for Himawari-8 (140.7E).
        """
        if date_time is None:
            date_time = self.get_latest_timestamp()
        
        # Format: YYYY/MM/DD/HHmm00_R_C.png
        # Level 4d (4x4 tiles). Singapore is roughly in Row 1, Col 1.
        level = "4d"
        tile_r = 1
        tile_c = 1
        
        date_str = date_time.strftime("%Y/%m/%d/%H%M00")
        # Update base URL to correct level structure
        base = self.base_url.replace("1d", level) 
        url = f"{base}/{date_str}_{tile_r}_{tile_c}.png"
        
        print(f"Fetching Satellite Tile (SGD Focus) from: {url}")
        
        try:
            # Disable SSL verification to avoid certificate errors on some macOS Python environments
            response = requests.get(url, verify=False, timeout=30)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Save with organized directory structure by year/month
                year_month_dir = os.path.join(self.save_dir, date_time.strftime("%Y"), date_time.strftime("%m"))
                os.makedirs(year_month_dir, exist_ok=True)
                
                filename = f"{year_month_dir}/himawari_{level}_{date_time.strftime('%Y%m%d_%H%M%S')}.png"
                img.save(filename)
                print(f"✓ Satellite image saved to {filename}")
                return filename
            else:
                print(f"✗ Failed to fetch image. Status: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Error fetching satellite image: {e}")
            return None
    
    def fetch_hourly_range(self, start_date, end_date=None, delay_seconds=1, daylight_only=True):
        """
        Fetch hourly satellite images for a date range.
        
        Args:
            start_date: datetime object or string in format 'YYYY-MM-DD HH:MM:SS' (UTC)
            end_date: datetime object or string in format 'YYYY-MM-DD HH:MM:SS' (UTC) (default: now)
            delay_seconds: delay between requests to avoid overwhelming the server
            daylight_only: if True, only fetch images during Singapore daylight hours (6am-6pm SGT)
        
        Returns:
            list of successfully downloaded filenames
        """
        # Parse start_date if string
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        
        # Parse or set end_date
        if end_date is None:
            end_date = self.get_latest_timestamp()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        start_date = start_date.replace(minute=0, second=0, microsecond=0)
        end_date = end_date.replace(minute=0, second=0, microsecond=0)
        
        print(f"\n{'='*60}")
        print(f"Fetching hourly data from {start_date} to {end_date} (UTC)")
        if daylight_only:
            print(f"Mode: DAYLIGHT ONLY (6am-6pm Singapore time)")
        else:
            print(f"Mode: ALL HOURS (including nighttime)")
        print(f"{'='*60}\n")
        
        downloaded_files = []
        failed_downloads = []
        skipped_night = []
        current_date = start_date
        total_hours = int((end_date - start_date).total_seconds() / 3600) + 1
        
        hour_count = 0
        processed_count = 0
        
        while current_date <= end_date:
            hour_count += 1
            
            # Check if it's nighttime in Singapore
            if daylight_only and not self.is_daylight_singapore(current_date):
                # Convert to Singapore time for display
                sg_time = pytz.UTC.localize(current_date).astimezone(self.singapore_tz)
                print(f"[{hour_count}/{total_hours}] Skipping (nighttime): {current_date} UTC = {sg_time.strftime('%Y-%m-%d %H:%M SGT')}")
                skipped_night.append(current_date)
                current_date += timedelta(hours=1)
                continue
            
            processed_count += 1
            sg_time = pytz.UTC.localize(current_date).astimezone(self.singapore_tz)
            print(f"\n[{hour_count}/{total_hours}] Processing (daylight): {current_date} UTC = {sg_time.strftime('%Y-%m-%d %H:%M SGT')}")
            
            filename = self.fetch_image(current_date)
            if filename:
                downloaded_files.append(filename)
            else:
                failed_downloads.append(current_date)
            
            current_date += timedelta(hours=1)
            
            if current_date <= end_date:
                time.sleep(delay_seconds)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  Total hours in range: {total_hours}")
        if daylight_only:
            print(f"  Skipped (nighttime): {len(skipped_night)}")
            print(f"  Daylight hours attempted: {processed_count}")
        else:
            print(f"  Attempted: {processed_count}")
        print(f"  Successfully downloaded: {len(downloaded_files)}")
        print(f"  Failed: {len(failed_downloads)}")
        print(f"{'='*60}\n")
        
        if failed_downloads:
            print("Failed downloads for timestamps:")
            for dt in failed_downloads[:10]:  # Show first 10
                sg_time = pytz.UTC.localize(dt).astimezone(self.singapore_tz)
                print(f"  - {dt} UTC ({sg_time.strftime('%Y-%m-%d %H:%M SGT')})")
            if len(failed_downloads) > 10:
                print(f"  ... and {len(failed_downloads) - 10} more")
        
        return downloaded_files

if __name__ == "__main__":
    collector = SatelliteCollector()
    
    collector.fetch_hourly_range(
        start_date='2024-01-01 00:00:00',
        end_date=None,          # defaults to latest available image time
        delay_seconds=1,
        daylight_only=True
    )