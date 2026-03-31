import json
import pandas as pd

# Load the forecast made earlier
with open('./forecast_latest.json') as f:
    forecast = json.load(f)

# Load historical data (which gets updated by your data pipeline)
df = pd.read_csv('./data/combined_dataset.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp')

print(f"\nForecast made at: {forecast['forecast_time_sgt']}")
print(f"\n{'Horizon':<10} {'Forecast':>12} {'Actual':>12} {'Error':>10}")
print("-" * 50)

for f in forecast['forecasts']:
    target_time = pd.to_datetime(f['time_sgt'])
    actual_row  = df[df['timestamp'] == target_time]
    
    if len(actual_row) > 0:
        actual_ghi = float(actual_row['ghi'].iloc[0])
        error      = abs(f['ghi_forecast_wm2'] - actual_ghi)
        print(f"  {f['horizon']:<8} {f['ghi_forecast_wm2']:>10.1f}  {actual_ghi:>10.1f}  {error:>8.1f} W/m²")
    else:
        print(f"  {f['horizon']:<8} {f['ghi_forecast_wm2']:>10.1f}  {'not yet':>10}  {'—':>10}")