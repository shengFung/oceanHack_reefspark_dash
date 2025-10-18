import pandas as pd

# Load data
df_temp = pd.read_csv("data/cleaned_surface_temp.csv")

print(f"Total temp rows: {len(df_temp)}")
print(f"Lat range: {df_temp['lat'].min():.2f} to {df_temp['lat'].max():.2f}")
print(f"Lon range: {df_temp['lon'].min():.2f} to {df_temp['lon'].max():.2f}")

# Test different region filters
region_bounds = {
    'global':     {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'tropics':    {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
    'arctic':     {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'antarctic':  {'lat_min': -90, 'lat_max': -66,'lon_min': -180, 'lon_max': 180},
    'indian':     {'lat_min': -30, 'lat_max': 30, 'lon_min': 20,   'lon_max': 120},
    'pacific':    {'lat_min': -30, 'lat_max': 30, 'lon_min': 120,  'lon_max': -100}
}

print("\n" + "="*60)
for region_name, bounds in region_bounds.items():
    if bounds['lon_min'] < bounds['lon_max']:
        filtered = df_temp[(df_temp['lat'].between(bounds['lat_min'], bounds['lat_max'])) & 
                          (df_temp['lon'].between(bounds['lon_min'], bounds['lon_max']))]
    else:
        filtered = df_temp[(df_temp['lat'].between(bounds['lat_min'], bounds['lat_max'])) & 
                          ((df_temp['lon'] >= bounds['lon_min']) | (df_temp['lon'] <= bounds['lon_max']))]
    
    if len(filtered) > 0:
        df_temp['time'] = pd.to_datetime(df_temp['time'])
        filtered['time'] = pd.to_datetime(filtered['time'])
        filtered['year'] = filtered['time'].dt.year
        yearly = filtered.groupby('year')['surface_temp'].mean()
        print(f"\n{region_name.upper()}:")
        print(f"  Rows after filter: {len(filtered)}")
        print(f"  Years: {yearly.index.min()} to {yearly.index.max()}")
        print(f"  Temp range: {yearly.min():.2f}°C to {yearly.max():.2f}°C")
        print(f"  Mean temp: {yearly.mean():.2f}°C")
    else:
        print(f"\n{region_name.upper()}: NO DATA")
