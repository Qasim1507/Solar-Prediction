import requests
import pandas as pd
import os
from datetime import datetime
import pvlib  # You'll need to install this: pip install pvlib

class HistoricalDataCollector:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.lat, self.lon = 1.3521, 103.8198
    
    def fetch_data(self, start_date="2024-01-01", end_date=None):
        if end_date is None:
            # Open-Meteo archive lags ~5 days behind realtime
            from datetime import timedelta
            end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"Fetching data from {start_date} to {end_date}...")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,cloud_cover,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
            "timezone": "Asia/Singapore"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            df = pd.DataFrame(response.json()['hourly'])
            
            # Clean and Rename
            df['time'] = pd.to_datetime(df['time'])
            df.rename(columns={'time': 'timestamp', 'shortwave_radiation': 'ghi'}, inplace=True)
            
            # --- STEP: ADD PV PHYSICS ---
            location = pvlib.location.Location(self.lat, self.lon, tz='Asia/Singapore')
            
            # Calculate Theoretical Clear Sky Radiation for these timestamps
            # FIX: Use timezone-aware datetime from the start
            times = pd.DatetimeIndex(df['timestamp'], tz='Asia/Singapore')
            
            clearsky = location.get_clearsky(times)
            df['ghi_clearsky'] = clearsky['ghi'].values
            
            # Simulate a 1kW PV System Output
            # Efficiency drops 0.4% per degree above 25°C (standard for SG)
            temp_coeff = -0.004 
            df['pv_power_predicted'] = df['ghi'] * (1 + temp_coeff * (df['temperature_2m'] - 25))
            df.loc[df['pv_power_predicted'] < 0, 'pv_power_predicted'] = 0
            
            # --- STEP: FILTER FOR DAYLIGHT ONLY ---
            # Remove rows where the sun is below the horizon
            df = df[df['ghi_clearsky'] > 0]
            
            filename = f"{self.save_dir}/pv_dataset_sg.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(df)} daylight rows to {filename}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Features: {', '.join(df.columns.tolist())}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    collector = HistoricalDataCollector()
    df = collector.fetch_data()
    
    if df is not None:
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Total records: {len(df)}")
        print(f"\nSample statistics:")
        print(df[['temperature_2m', 'cloud_cover', 'ghi', 'pv_power_predicted']].describe())
        print("\nFirst few rows:")
        print(df.head())