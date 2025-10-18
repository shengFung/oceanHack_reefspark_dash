from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def _load_litter_df():
    df = pd.read_csv("data/litter_dataset_final_data.csv")
    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/litters')
def litters_page():
    return render_template('litters.html')

@app.route('/simek')
def simek_page():
    return render_template('simek.html')

@app.route('/live-monitor')
def live_monitor():
    return render_template('live_monitor.html')

@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')

@app.route('/api/forecast/trends')
def forecast_trends():
    """Generate forecasts for temperature, oxygen, and salinity trends."""
    df_temp = pd.read_csv("data/cleaned_surface_temp.csv")
    df_oxy = pd.read_csv("data/cleaned_surface_oxy.csv")
    df_sal = pd.read_csv("data/cleaned_surface_sal.csv")

    region = request.args.get('region', 'global')
    forecast_years = int(request.args.get('forecast_years', 5))

    # Parse timestamps
    for df in [df_temp, df_oxy, df_sal]:
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year

    # Region filtering based on bounds
    region_bounds = {
        'global':     {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'tropics':    {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
        'arctic':     {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'antarctic':  {'lat_min': -90, 'lat_max': -66,'lon_min': -180, 'lon_max': 180},
        'indian':     {'lat_min': -30, 'lat_max': 30, 'lon_min': 20,   'lon_max': 120},
        'pacific':    {'lat_min': -30, 'lat_max': 30, 'lon_min': 120,  'lon_max': -100}
    }
    b = region_bounds.get(region, region_bounds['global'])

    # Filter data by region
    for df in [df_temp, df_oxy, df_sal]:
        if b['lon_min'] < b['lon_max']:
            df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                    (df['lon'].between(b['lon_min'], b['lon_max']))]
        else:
            df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                    ((df['lon'] >= b['lon_min']) | (df['lon'] <= b['lon_max']))]

    # Work with historical data (1800-1940)
    df_temp = df_temp[(df_temp['year'] >= 1800) & (df_temp['year'] <= 1940)]
    df_oxy = df_oxy[(df_oxy['year'] >= 1800) & (df_oxy['year'] <= 1940)]
    df_sal = df_sal[(df_sal['year'] >= 1800) & (df_sal['year'] <= 1940)]
    
    # For analysis, we'll use the complete historical range
    temp_start_year = df_temp['year'].min()
    temp_end_year = df_temp['year'].max()
    oxy_start_year = df_oxy['year'].min()
    oxy_end_year = df_oxy['year'].max()
    
    # Calculate yearly averages
    yearly_temp = df_temp.groupby('year')['surface_temp'].mean().reset_index()
    yearly_oxy = df_oxy.groupby('year')['oxygen_mg_L'].mean().reset_index()
    yearly_sal = df_sal.groupby('year')['surface_sal'].mean().reset_index()

    # Generate forecasts using linear regression
    def generate_forecast(data, periods, window_size=15):
            """Generate forecast with smooth transition from historical data."""
            # Prepare data
            if len(data) > window_size:
                historical_data = data[-window_size:]  # Use last window_size points
                X = np.array(range(len(historical_data))).reshape(-1, 1)
                y = historical_data.values
            else:
                X = np.array(range(len(data))).reshape(-1, 1)
                y = data.values

            # Fit model on historical data
            model = LinearRegression()
            model.fit(X, y)

            # Get last historical value and trend
            last_value = y[-1]
            trend = model.coef_[0]

            # Generate future points
            forecast = []
            historical_std = np.std(y)
        
            for i in range(periods):
                # Calculate base prediction
                next_value = last_value + trend * (i + 1)
            
                # Add decreasing noise based on historical variance
                noise_scale = historical_std * 0.1 * (1 - i/periods)  # Reduce noise over time
                noise = np.random.normal(0, noise_scale)
            
                forecast.append(next_value + noise)
                last_value = next_value  # Update for next iteration

            return forecast

    # Generate future years for forecasting
    last_year = max(yearly_temp['year'].max(),
                    yearly_oxy['year'].max(),
                    yearly_sal['year'].max())
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))

    # Create forecasts
    temp_forecast = generate_forecast(yearly_temp['surface_temp'], forecast_years)
    oxy_forecast = generate_forecast(yearly_oxy['oxygen_mg_L'], forecast_years)
    sal_forecast = generate_forecast(yearly_sal['surface_sal'], forecast_years)

    return jsonify({
        'historical': {
            'years': yearly_temp['year'].tolist(),
            'temperature': yearly_temp['surface_temp'].round(2).tolist(),
            'oxygen': yearly_oxy['oxygen_mg_L'].round(2).tolist(),
            'salinity': yearly_sal['surface_sal'].round(2).tolist()
        },
        'forecast': {
            'years': future_years,
            'temperature': [round(x, 2) for x in temp_forecast],
            'oxygen': [round(x, 2) for x in oxy_forecast],
            'salinity': [round(x, 2) for x in sal_forecast]
        }
    })


@app.route('/api/forecast/rsi')
def forecast_rsi():
    """Generate forecast for Reef Stress Index (RSI) components."""
    region = request.args.get('region', 'global')
    forecast_years = int(request.args.get('forecast_years', 5))
    
    # Get historical RSI data first
    rsi_response = reef_stress()
    rsi_data = json.loads(rsi_response.get_data(as_text=True))
    
    # Prepare data for forecasting (use last N points to compute trend)
    def make_smooth_forecast(values, periods, window=10):
        arr = np.array(values, dtype=float)
        if len(arr) >= window:
            y = arr[-window:]
        else:
            y = arr

        # x as 0..n-1
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slope = float(model.coef_[0])
        last_val = float(y[-1])
        hist_std = float(np.std(y)) if len(y) > 1 else 0.0

        forecasts = []
        current = last_val
        for i in range(periods):
            # predict next by linear step
            next_base = current + slope
            # noise scaled to historical variability and horizon
            noise = np.random.normal(0, hist_std * 0.1 * (1 + i / periods))
            next_val = next_base + noise
            # clip to [0,1]
            next_val = float(np.clip(next_val, 0.0, 1.0))
            forecasts.append(next_val)
            current = next_base

        return forecasts

    years = rsi_data['years']
    rsi_values = rsi_data['RSI']
    tsi_values = rsi_data['TSI']
    osi_values = rsi_data['OSI']

    if not years or not rsi_values or not tsi_values or not osi_values:
        return jsonify({
            'historical': rsi_data,
            'forecast': {
                'years': [],
                'RSI': [],
                'TSI': [],
                'OSI': []
            }
        })

    last_year = int(years[-1])
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))

    rsi_forecast = make_smooth_forecast(rsi_values, forecast_years, window=10)
    tsi_forecast = make_smooth_forecast(tsi_values, forecast_years, window=10)
    osi_forecast = make_smooth_forecast(osi_values, forecast_years, window=10)
    
    return jsonify({
        'historical': {
            'years': rsi_data['years'],
            'RSI': rsi_data['RSI'],
            'TSI': rsi_data['TSI'],
            'OSI': rsi_data['OSI']
        },
        'forecast': {
            'years': future_years,  # <-- FIXED
            'RSI': [round(x, 3) for x in rsi_forecast],  # <-- FIXED
            'TSI': [round(x, 3) for x in tsi_forecast],  # <-- FIXED
            'OSI': [round(x, 3) for x in osi_forecast]   # <-- FIXED
        }
    })


@app.route('/api/litters/trends')
def litters_trends():
    """Return yearly totals and averages across all beaches.
    Output: { years: [...], total_abund: [...], avg_abund_per_survey: [...] }
    """
    df = _load_litter_df()
    
    # optional filters
    country = request.args.get('country')
    search = request.args.get('search')
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]
    
    # collect pairs like 2012_abund / 2012_nbsur ... 2023
    years = []
    totals = []
    avgs = []
    for year in range(2012, 2024):
        abund_col = f"{year}_abund"
        nbsur_col = f"{year}_nbsur"
        if abund_col in df.columns and nbsur_col in df.columns:
            total_abund = pd.to_numeric(df[abund_col], errors='coerce').sum(min_count=1)
            total_survey = pd.to_numeric(df[nbsur_col], errors='coerce').sum(min_count=1)
            avg_per_survey = (total_abund / total_survey) if total_survey and total_survey != 0 else None
            years.append(year)
            totals.append(round(float(total_abund), 2) if pd.notna(total_abund) else None)
            avgs.append(round(float(avg_per_survey), 3) if avg_per_survey is not None and pd.notna(avg_per_survey) else None)
    
    return jsonify({"years": years, "total_abund": totals, "avg_abund_per_survey": avgs})

@app.route('/api/litters/predictions')
def litters_predictions():
    """Predict next 5 years litter for each beach using slope/intercept columns.
    Query: country, search, sort, order, page, pageSize
    Output: rows with beach info and predicted litter for 2024–2028
    """
    df = _load_litter_df()
    country = request.args.get('country')
    search = request.args.get('search')
    sort = request.args.get('sort', 'pred2028')
    order = request.args.get('order', 'desc')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 10))

    # filter
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # predict for 2024–2028 using y = slope * year + intercept
    # If slope/intercept missing, fallback to last available year
    years = list(range(2024, 2029))
    preds = []
    for _, r in df.iterrows():
        slope = r.get('litter_slope')
        intercept = r.get('litter_intercept')
        try:
            slope = float(slope)
            intercept = float(intercept)
        except (TypeError, ValueError):
            slope = None
            intercept = None
        pred_row = {
            'country': str(r.get('country', '')),
            'beachname': str(r.get('beachname', '')),
            'beachcode': str(r.get('beachcode', '')),
            'slope': slope,
            'intercept': intercept,
            'preds': []
        }
        for year in years:
            if slope is not None and intercept is not None:
                pred = slope * year + intercept
            else:
                # fallback: use last available year
                last_year = 2023
                pred = r.get(f'{last_year}_abund', None)
                try:
                    pred = float(pred)
                except (TypeError, ValueError):
                    pred = None
            pred_row['preds'].append(round(pred, 2) if pred is not None else None)
        # for sorting
        pred_row['pred2028'] = pred_row['preds'][-1] if pred_row['preds'] else None
        preds.append(pred_row)

    # sort
    sort_map = {
        'pred2028': 'pred2028',
        'name': 'beachname',
        'code': 'beachcode',
        'country': 'country'
    }
    sort_col = sort_map.get(sort, 'pred2028')
    preds = sorted(preds, key=lambda x: (x.get(sort_col) is None, x.get(sort_col)), reverse=(order=='desc'))

    # paginate
    total_rows = len(preds)
    start = (page - 1) * page_size
    end = start + page_size
    page_preds = preds[start:end]

    return jsonify({
        'total': int(total_rows),
        'page': page,
        'pageSize': page_size,
        'years': years,
        'rows': page_preds
    })

    # collect pairs like 2012_abund / 2012_nbsur ... 2023
    years = []
    totals = []
    avgs = []
    # optional filters
    country = request.args.get('country')
    search = request.args.get('search')  # matches in beachname or beachcode
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    for year in range(2012, 2024):
        abund_col = f"{year}_abund"
        nbsur_col = f"{year}_nbsur"
        if abund_col in df.columns and nbsur_col in df.columns:
            total_abund = pd.to_numeric(df[abund_col], errors='coerce').sum(min_count=1)
            total_survey = pd.to_numeric(df[nbsur_col], errors='coerce').sum(min_count=1)
            avg_per_survey = (total_abund / total_survey) if total_survey and total_survey != 0 else None
            years.append(year)
            totals.append(round(float(total_abund), 2) if pd.notna(total_abund) else None)
            avgs.append(round(float(avg_per_survey), 3) if avg_per_survey is not None and pd.notna(avg_per_survey) else None)

    return jsonify({"years": years, "total_abund": totals, "avg_abund_per_survey": avgs})

@app.route('/api/litters/top_beaches')
def litters_top_beaches():
    """Top N beaches by totalLitter (default 10)."""
    N = int(request.args.get('n', 10))
    country = request.args.get('country')
    search = request.args.get('search')
    df = _load_litter_df()
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if 'totalLitter' in df.columns:
        df['totalLitter'] = pd.to_numeric(df['totalLitter'], errors='coerce')
        top = df.sort_values('totalLitter', ascending=False).head(N)
        return jsonify({
            "labels": top.get('beachname', top.get('beachcode', top.index)).astype(str).tolist(),
            "values": top['totalLitter'].round(2).fillna(0).tolist()
        })
    # fallback: compute total across yearly columns
    yearly_cols = [c for c in df.columns if c.endswith('_abund')]
    df['computed_total'] = df[yearly_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    top = df.sort_values('computed_total', ascending=False).head(N)
    return jsonify({
        "labels": top.get('beachname', top.get('beachcode', top.index)).astype(str).tolist(),
        "values": top['computed_total'].round(2).fillna(0).tolist()
    })

@app.route('/api/litters/countries')
def litters_countries():
    df = _load_litter_df()
    countries = sorted(set(df.get('country', pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
    return jsonify({"countries": countries})

@app.route('/api/litters/rows')
def litters_rows():
    """Return paged/sorted rows with computed totals for UI table.
    Query: country, search, sort (total|avg|name|code), order (asc|desc), page, pageSize
    """
    df = _load_litter_df()
    country = request.args.get('country')
    search = request.args.get('search')
    sort = request.args.get('sort', 'total')
    order = request.args.get('order', 'desc')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 20))

    # filter
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # totals/averages
    yearly_abund = [c for c in df.columns if c.endswith('_abund')]
    yearly_nbsur = [c for c in df.columns if c.endswith('_nbsur')]
    df['total_abund'] = df[yearly_abund].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    df['total_survey'] = df[yearly_nbsur].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    df['avg_per_survey'] = df.apply(lambda r: (r['total_abund'] / r['total_survey']) if pd.notna(r['total_abund']) and pd.notna(r['total_survey']) and r['total_survey'] != 0 else None, axis=1)

    # sort
    sort_map = {
        'total': 'total_abund',
        'avg': 'avg_per_survey',
        'name': 'beachname',
        'code': 'beachcode'
    }
    sort_col = sort_map.get(sort, 'total_abund')
    df = df.sort_values(sort_col, ascending=(order == 'asc'), na_position='last')

    # paginate
    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    rows = []
    for _, r in page_df.iterrows():
        rows.append({
            'country': str(r.get('country', '')),
            'beachname': str(r.get('beachname', '')),
            'beachcode': str(r.get('beachcode', '')),
            'total_abund': None if pd.isna(r.get('total_abund')) else float(r.get('total_abund')),
            'total_survey': None if pd.isna(r.get('total_survey')) else float(r.get('total_survey')),
            'avg_per_survey': None if pd.isna(r.get('avg_per_survey')) else float(r.get('avg_per_survey'))
        })

    return jsonify({
        'total': int(total_rows),
        'page': page,
        'pageSize': page_size,
        'rows': rows
    })

@app.route('/api/trends')
def trends():
    df_temp = pd.read_csv("data/cleaned_surface_temp.csv")
    df_oxy = pd.read_csv("data/cleaned_surface_oxy.csv")
    df_sal = pd.read_csv("data/cleaned_surface_sal.csv")

    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp['year'] = df_temp['time'].dt.year

    df_oxy['time'] = pd.to_datetime(df_oxy['time'])
    df_oxy['year'] = df_oxy['time'].dt.year

    df_sal['time'] = pd.to_datetime(df_sal['time'])
    df_sal['year'] = df_sal['time'].dt.year

    # Get region from query string
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    # Filter by selected region
    df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
    df_oxy = df_oxy[(df_oxy['lat'].between(lat_min, lat_max)) & (df_oxy['lon'].between(lon_min, lon_max))]
    df_sal = df_sal[(df_sal['lat'].between(lat_min, lat_max)) & (df_sal['lon'].between(lon_min, lon_max))]

    yearly_temp = df_temp.groupby('year').agg({
        'surface_temp': 'mean'
    }).reset_index()

    yearly_oxy = df_oxy.groupby('year').agg({
        'oxygen_mg_L': 'mean'
    }).reset_index()

    yearly_sal = df_sal.groupby('year').agg({
        'surface_sal': 'mean'
    }).reset_index()

    return jsonify({
        'labels': yearly_temp['year'].tolist(),
        'temp_data': yearly_temp['surface_temp'].round(2).tolist(),
        'labels2': yearly_oxy['year'].tolist(),
        'oxy_data': yearly_oxy['oxygen_mg_L'].round(2).tolist(),
        'labels3': yearly_sal['year'].tolist(),
        'sal_data': yearly_sal['surface_sal'].round(2).tolist()
    })

@app.route('/api/temp_oxy_correlation')
def temp_oxy_correlation():
    df_temp = pd.read_csv("data/cleaned_surface_temp.csv")
    df_oxy = pd.read_csv("data/cleaned_surface_oxy.csv")

    # Region filtering
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp['year'] = df_temp['time'].dt.year
    df_oxy['time'] = pd.to_datetime(df_oxy['time'])
    df_oxy['year'] = df_oxy['time'].dt.year

    # Filter by region
    df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
    df_oxy = df_oxy[(df_oxy['lat'].between(lat_min, lat_max)) & (df_oxy['lon'].between(lon_min, lon_max))]

    # Merge & calculate correlation
    df = pd.merge(df_temp, df_oxy, on=['time', 'lat', 'lon'])
    df = df.dropna(subset=['surface_temp', 'oxygen_mg_L'])

    # FIX: Extract year again after merge
    df['year'] = pd.to_datetime(df['time']).dt.year

    corr_by_year = (
        df.groupby('year')
        .apply(lambda g: g['surface_temp'].corr(g['oxygen_mg_L']))
        .dropna()
        .reset_index(name='correlation')
    )

    return jsonify({
        'years': corr_by_year['year'].tolist(),
        'correlation': corr_by_year['correlation'].round(2).tolist()
    })



# @app.route('/api/ts_diagram')
# def ts_diagram():
#     import pandas as pd

#     df_temp = pd.read_csv("data/vardepth_temp.csv")
#     df_sal = pd.read_csv("data/vardepth_sal.csv")

#     # Region filtering
#     lat_min = float(request.args.get('lat_min', -90))
#     lat_max = float(request.args.get('lat_max', 90))
#     lon_min = float(request.args.get('lon_min', -180))
#     lon_max = float(request.args.get('lon_max', 180))

#     df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
#     df_sal = df_sal[(df_sal['lat'].between(lat_min, lat_max)) & (df_sal['lon'].between(lon_min, lon_max))]

#     # Merge on time, lat, lon, z (depth)
#     df_temp['time'] = pd.to_datetime(df_temp['time'])
#     df_sal['time'] = pd.to_datetime(df_sal['time'])
#     df = pd.merge(df_temp, df_sal, on=['time', 'lat', 'lon', 'z'])

#     df = df.dropna(subset=['temp', 'salinity', 'z'])
#     df = df.sample(n=min(1000, len(df)))  # Limit to 1000 points for performance

#     return jsonify({
#         'temp': df['temp'].round(2).tolist(),
#         'sal': df['salinity'].round(2).tolist(),
#         'depth': df['z'].round(1).tolist()
#     })





@app.route('/api/ts_scatter_by_year')
def ts_scatter_by_year():
    year = int(request.args.get('year'))
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df_temp = pd.read_csv("data/vardepth_temp.csv")
    df_sal = pd.read_csv("data/vardepth_salinity.csv")

    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_sal['time'] = pd.to_datetime(df_sal['time'])

    df_temp['year'] = df_temp['time'].dt.year
    df_sal['year'] = df_sal['time'].dt.year


    df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
    df_sal = df_sal[(df_sal['year'] == year) & (df_sal['lat'].between(lat_min, lat_max)) & (df_sal['lon'].between(lon_min, lon_max))]

    for df in [df_temp, df_sal]:
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df['z'] = df['z'].round(1)


    df = pd.merge(df_temp, df_sal, on=['time', 'lat', 'lon', 'z'])
    df = df.dropna(subset=['temp', 'salinity'])

    return jsonify({
    'temp': df['temp'].tolist(),
    'sal': df['salinity'].tolist(),
    'depth': df['z'].tolist()  
    })



@app.route('/api/strat_proxy_by_year')
def strat_proxy_by_year():
    import pandas as pd
    from flask import request, jsonify

    year = int(request.args.get('year'))
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df = pd.read_csv("data/vardepth_temp.csv")
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df = df[df['year'] == year]
    df = df[(df['lat'].between(lat_min, lat_max)) & (df['lon'].between(lon_min, lon_max))]
    df = df.dropna(subset=['temp', 'z'])

    # Round for merge stability
    df['lat'] = df['lat'].round(2)
    df['lon'] = df['lon'].round(2)
    df['z'] = df['z'].round(1)

    # Pivot temperature values by depth
    df_surface = df[df['z'] == 0.0].rename(columns={'temp': 'temp_surface'})
    df_200m = df[df['z'] == 200.0].rename(columns={'temp': 'temp_200m'})

    merged = pd.merge(df_surface, df_200m, on=['time', 'lat', 'lon'], how='inner')

    merged['delta_T'] = merged['temp_surface'] - merged['temp_200m']

    # print(merged[['lat', 'lon', 'delta_T']].head())

    return jsonify({
        'lat': merged['lat'].tolist(),
        'lon': merged['lon'].tolist(),
        'delta_T': merged['delta_T'].round(2).tolist()
    })


@app.route('/api/climatology_anomalies')
def climatology_anomalies():
    df_temp = pd.read_csv("data/cleaned_surface_temp.csv")

    # Parse time
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp['year'] = df_temp['time'].dt.year
    df_temp['month'] = df_temp['time'].dt.month

    # Region filtering
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    # Handle Pacific special case (wrap dateline)
    if lon_min > lon_max:
        df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) &
                        ((df_temp['lon'] >= lon_min) | (df_temp['lon'] <= lon_max))]
    else:
        df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) &
                        (df_temp['lon'].between(lon_min, lon_max))]

    df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]

    # --- Climatology (mean per month across all years) ---
    climatology = (
        df_temp.groupby('month')['surface_temp']
        .mean()
        .reset_index(name='climatology')
    )

    # --- Monthly means per year (for anomaly calc) ---
    monthly_means = (
        df_temp.groupby(['year', 'month'])['surface_temp']
        .mean()
        .reset_index()
    )

    # Merge with climatology
    merged = pd.merge(monthly_means, climatology, on='month')
    merged['anomaly'] = merged['surface_temp'] - merged['climatology']

    return jsonify({
        "months": climatology['month'].tolist(),
        "climatology": climatology['climatology'].round(2).tolist(),
        "years": merged['year'].tolist(),
        "monthly_temp": merged['surface_temp'].round(2).tolist(),
        "anomaly": merged['anomaly'].round(2).tolist()
    })


@app.route('/api/vertical_profile')
def vertical_profile():
    year = int(request.args.get('year'))
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    region = request.args.get('region')  # new: "global", "tropics", etc.
    tol = 0.5

    # Load datasets
    df_temp = pd.read_csv("data/vardepth_temp.csv")
    df_oxy = pd.read_csv("data/vardepth_oxy.csv")

    # Parse time
    for df in [df_temp, df_oxy]:
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year

    # --- REGION FILTERING ---
    if region:
        if region == "global":
            df_temp = df_temp[df_temp['year'] == year]
            df_oxy = df_oxy[df_oxy['year'] == year]
        elif region == "tropics":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'].between(-23, 23))]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'].between(-23, 23))]
        elif region == "arctic":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'] >= 66)]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'] >= 66)]
        elif region == "antarctic":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'] <= -66)]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'] <= -66)]
        elif region == "indian":
            df_temp = df_temp[(df_temp['year'] == year) &
                              (df_temp['lat'].between(-30, 30)) &
                              (df_temp['lon'].between(20, 120))]
            df_oxy = df_oxy[(df_oxy['year'] == year) &
                            (df_oxy['lat'].between(-30, 30)) &
                            (df_oxy['lon'].between(20, 120))]
        elif region == "pacific":
            df_temp = df_temp[(df_temp['year'] == year) &
                              (df_temp['lat'].between(-30, 30)) &
                              (df_temp['lon'].between(120, -100))]  # adjust wrap if needed
            df_oxy = df_oxy[(df_oxy['year'] == year) &
                            (df_oxy['lat'].between(-30, 30)) &
                            (df_oxy['lon'].between(120, -100))]
    else:
        # --- POINT FILTERING (default) ---
        lat = float(lat)
        lon = float(lon)
        df_temp = df_temp[(df_temp['year'] == year) &
                          (df_temp['lat'].between(lat - tol, lat + tol)) &
                          (df_temp['lon'].between(lon - tol, lon + tol))]
        df_oxy = df_oxy[(df_oxy['year'] == year) &
                        (df_oxy['lat'].between(lat - tol, lat + tol)) &
                        (df_oxy['lon'].between(lon - tol, lon + tol))]

    # Round coords & merge
    for df in [df_temp, df_oxy]:
        if not df.empty:
            df['lat'] = df['lat'].round(2)
            df['lon'] = df['lon'].round(2)
            df['z'] = df['z'].round(1)

    df = pd.merge(df_temp, df_oxy, on=['time','lat','lon','z'], how='inner')
    if df.empty:
        return jsonify({"depth": [], "temp": [], "oxygen": []})

    df = df.dropna(subset=['temp','oxy'])

    # Average profile across casts
    profile = df.groupby('z').agg({
        'temp': 'mean',
        'oxy': 'mean'
    }).reset_index()

    return jsonify({
        "depth": profile['z'].tolist(),
        "temp": profile['temp'].round(2).tolist(),
        "oxygen": profile['oxy'].round(2).tolist()
    })


@app.route('/api/hovmoller')
def hovmoller():
    import numpy as np
    import pandas as pd
    from flask import request, jsonify

    variable = request.args.get('var', 'temp')
    region = request.args.get('region', 'global')

    region_bounds = {
        'global':     {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'tropics':    {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
        'arctic':     {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'antarctic':  {'lat_min': -90, 'lat_max': -66, 'lon_min': -180, 'lon_max': 180},
        'indian':     {'lat_min': -30, 'lat_max': 30, 'lon_min': 20,  'lon_max': 120},
        'pacific':    {'lat_min': -30, 'lat_max': 30, 'lon_min': 120, 'lon_max': -100},
    }
    b = region_bounds.get(region, region_bounds['global'])

    df = pd.read_csv(f"data/vardepth_{variable}.csv")
    df = df.dropna(subset=['lat', 'lon', 'z', variable])
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['year'] = df['time'].dt.year

    # Handle Pacific
    if b['lon_min'] < b['lon_max']:
        df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                (df['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                ((df['lon'] >= b['lon_min']) | (df['lon'] <= b['lon_max']))]

    df_mean = df.groupby(['year', 'z'])[variable].mean().reset_index()
    grid = df_mean.pivot(index='z', columns='year', values=variable)

    # Fill missing with None safely
    grid = grid.replace([np.nan, np.inf, -np.inf], None)

    return jsonify({
        "year": [int(y) for y in grid.columns.tolist()],
        "depth": [float(z) for z in grid.index.tolist()],
        "values": [[None if (pd.isna(v) or v is np.nan) else float(v) for v in row] for row in grid.values]
    })



@app.route('/api/reef_stress')
def reef_stress():
    """
    Returns yearly Reef Stress Index (RSI) for a selected region.
    Query params:
      - region (default 'global')
      - start (optional, int year)
      - end (optional, int year)
      - w_tsi, w_hci, w_osi (optional weights, default 0.5,0.3,0.2)
    Response:
      {
        "years": [...],
        "TSI": [...],      # thermal stress index (0-1)
        "HCI": [...],      # stratification / heat content index  (0-1)
        "OSI": [...],      # oxygen stress index (0-1)
        "RSI": [...],      # combined index (0-1)
      }
    """
    import numpy as np
    import pandas as pd
    from flask import request, jsonify

    # --- user params ---
    region = request.args.get('region', 'global')
    start = request.args.get('start', None)
    end = request.args.get('end', None)
    # weights (must sum to 1 ideally)
    w_tsi = float(request.args.get('w_tsi', 0.5))
    w_hci = float(request.args.get('w_hci', 0.3))
    w_osi = float(request.args.get('w_osi', 0.2))

    # safe normalization of weights
    total_w = (w_tsi + w_hci + w_osi)
    if total_w == 0:
        w_tsi, w_hci, w_osi = 0.5, 0.3, 0.2
    else:
        w_tsi, w_hci, w_osi = w_tsi/total_w, w_hci/total_w, w_osi/total_w

    # region bounds
    region_bounds = {
        'global':  {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'tropics': {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
        'arctic':  {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'antarctic': {'lat_min': -90, 'lat_max': -66, 'lon_min': -180, 'lon_max': 180},
        'indian':  {'lat_min': -30, 'lat_max': 30, 'lon_min': 20, 'lon_max': 120},
        # pacific spans dateline: we'll handle wrap-around below
        'pacific': {'lat_min': -30, 'lat_max': 30, 'lon_min': 120, 'lon_max': -100}
    }
    b = region_bounds.get(region, region_bounds['global'])

    # --- load data (depth datasets) ---
    df_temp = pd.read_csv("data/vardepth_temp.csv")
    df_oxy  = pd.read_csv("data/vardepth_oxy.csv")

    # basic cleaning/parsing
    for df in (df_temp, df_oxy):
        df.dropna(subset=['time', 'lat', 'lon', 'z'], inplace=True)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['time'], inplace=True)
        df['year'] = df['time'].dt.year

    # filter region (handle pacific dateline wrap)
    if b['lon_min'] < b['lon_max']:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          (df_temp['lon'].between(b['lon_min'], b['lon_max']))]
        df_oxy  = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                         (df_oxy['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          ((df_temp['lon'] >= b['lon_min']) | (df_temp['lon'] <= b['lon_max']))]
        df_oxy  = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                         ((df_oxy['lon'] >= b['lon_min']) | (df_oxy['lon'] <= b['lon_max']))]

    # optional year window
    if start:
        start = int(start)
        df_temp = df_temp[df_temp['year'] >= start]
        df_oxy  = df_oxy[df_oxy['year'] >= start]
    if end:
        end = int(end)
        df_temp = df_temp[df_temp['year'] <= end]
        df_oxy  = df_oxy[df_oxy['year'] <= end]

    # --- COMPONENT 1: TSI (Thermal Stress Index) ---
    # Use surface values only (z == 0.0) when available. If exact 0 missing, use nearest shallow depth (z <= 5m)
    def select_surface(df):
        if (df['z'] == 0.0).any():
            return df[df['z'] == 0.0].copy()
        else:
            return df[df['z'] <= 5.0].copy()

    surf_temp = select_surface(df_temp)
    surf_oxy  = select_surface(df_oxy)

    # compute climatology (long-term mean surface temp across all years in this region)
    if surf_temp.empty:
        return jsonify({"years": [], "TSI": [], "HCI": [], "OSI": [], "RSI": []})
    clim_mean = surf_temp.groupby('month' if 'month' in surf_temp.columns else surf_temp['time'].dt.month)['temp'].mean() \
                if 'month' in surf_temp.columns else None

    # Simpler: compute long-term mean and std at surface across all years
    longmean_temp = surf_temp['temp'].mean()
    longstd_temp  = surf_temp['temp'].std(ddof=0) if surf_temp['temp'].std(ddof=0) > 0 else 1.0

    # compute yearly mean surface temp and anomaly standardized
    yearly_surf_temp = surf_temp.groupby('year')['temp'].mean().reset_index(name='temp_mean')
    yearly_surf_temp['TSI_raw'] = (yearly_surf_temp['temp_mean'] - longmean_temp) / longstd_temp
    # convert TSI_raw to 0-1 by mapping -3..+3 sigma -> 0..1 (clamp)
    yearly_surf_temp['TSI'] = yearly_surf_temp['TSI_raw'].clip(-3, 3).apply(lambda x: (x + 3) / 6.0)

    # --- COMPONENT 2: HCI (Heat / Stratification Index) ---
    # For each cast-date find temp at z==0 and z==200 (or nearest). Then compute delta = temp_surface - temp_200
    # We'll compute yearly mean of delta across casts
    df_temp['z_round'] = df_temp['z'].round(0)
    # surface temp per cast
    surf = df_temp[df_temp['z_round'] == 0].rename(columns={'temp': 'temp_surface'})[['time','lat','lon','year','temp_surface']]
    depth200 = df_temp[df_temp['z_round'] == 200].rename(columns={'temp': 'temp_200'})[['time','lat','lon','year','temp_200']]

    # merge on time/lat/lon/year to get pairs; use inner merge
    merged200 = pd.merge(surf, depth200, on=['time','lat','lon','year'], how='inner')
    if merged200.empty:
        # fallback: compute surface minus mean of deeper layer (e.g., mean temp between 150-250m)
        deeper = df_temp[(df_temp['z'] >= 150) & (df_temp['z'] <= 250)]
        deeper_mean = deeper.groupby(['time','lat','lon','year'])['temp'].mean().reset_index(name='temp_200')
        merged200 = pd.merge(surf, deeper_mean, on=['time','lat','lon','year'], how='inner')

    if merged200.empty:
        # if still empty, HCI zeros
        yearly_hci = pd.DataFrame({'year': yearly_surf_temp['year'], 'HCI': 0.0})
    else:
        merged200['delta_T'] = merged200['temp_surface'] - merged200['temp_200']
        yearly_hci = merged200.groupby('year')['delta_T'].mean().reset_index(name='delta_mean')
        # normalize to 0-1 by clipping to plausible range e.g., 0..10 degC
        ymin, ymax = 0.0, 10.0
        yearly_hci['HCI'] = yearly_hci['delta_mean'].clip(ymin, ymax).apply(lambda x: (x - ymin) / (ymax - ymin))

    # --- COMPONENT 3: OSI (Oxygen Stress Index) ---
    # Use surface oxygen mean per year: low oxygen -> higher stress.
    yearly_surf_oxy = surf_oxy.groupby('year')['oxy'].mean().reset_index(name='oxy_mean')
    # We'll convert oxygen into stress index: high O2 -> low stress; so invert and normalize.
    # Determine realistic bounds (example): oxy 0..10 mg/L (0 worst, 10 best)
    oxy_min, oxy_max = 0.0, 10.0
    yearly_surf_oxy['oxy_clip'] = yearly_surf_oxy['oxy_mean'].clip(oxy_min, oxy_max)
    yearly_surf_oxy['OSI'] = yearly_surf_oxy['oxy_clip'].apply(lambda x: 1.0 - ((x - oxy_min) / (oxy_max - oxy_min)))

    # --- MERGE yearly components into a single table ---
    years = sorted(set(yearly_surf_temp['year'].tolist() +
                       yearly_hci['year'].tolist() +
                       yearly_surf_oxy['year'].tolist()))

    df_yearly = pd.DataFrame({'year': years})
    df_yearly = df_yearly.merge(yearly_surf_temp[['year','TSI']], on='year', how='left')
    df_yearly = df_yearly.merge(yearly_hci[['year','HCI']], on='year', how='left')
    df_yearly = df_yearly.merge(yearly_surf_oxy[['year','OSI']], on='year', how='left')

    # fill missing component values conservatively
    df_yearly['TSI'] = df_yearly['TSI'].fillna(0.5)  # neutral
    df_yearly['HCI'] = df_yearly['HCI'].fillna(0.0)
    df_yearly['OSI'] = df_yearly['OSI'].fillna(0.5)

    # Combined RSI (0 low stress -> 1 high stress)
    df_yearly['RSI_raw'] = w_tsi * df_yearly['TSI'] + w_hci * df_yearly['HCI'] + w_osi * df_yearly['OSI']
    # Clip to [0,1] to be safe
    df_yearly['RSI'] = df_yearly['RSI_raw'].clip(0.0, 1.0)

    # filter start/end if explicitly provided & ensure sorted
    if start:
        df_yearly = df_yearly[df_yearly['year'] >= start]
    if end:
        df_yearly = df_yearly[df_yearly['year'] <= end]
    df_yearly = df_yearly.sort_values('year')

    return jsonify({
        "years": df_yearly['year'].astype(int).tolist(),
        "TSI": df_yearly['TSI'].round(3).tolist(),
        "HCI": df_yearly['HCI'].round(3).tolist(),
        "OSI": df_yearly['OSI'].round(3).tolist(),
        "RSI": df_yearly['RSI'].round(3).tolist(),
        "weights": {"w_tsi": w_tsi, "w_hci": w_hci, "w_osi": w_osi}
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that uses Google Gemini AI to answer questions about environmental data
    and make predictions for reef health insights.
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Load environmental dataset for context
        df = _load_litter_df()
        
        # Create dataset summary for context
        dataset_summary = f"""
Dataset Information:
- Total beaches: {len(df)}
- Countries: {', '.join(df['country'].unique())}
- Years: 2012-2023 (yearly abundance and survey counts)
- Total litter items recorded: {df['totalLitter'].sum():,.0f}
- Average litter per beach: {df['avgLitter'].mean():.2f}
- Clusters: {', '.join(map(str, df['cluster'].unique()))}

Key columns:
- YYYY_abund: Litter abundance for year YYYY
- YYYY_nbsur: Number of surveys for year YYYY
- totalLitter: Total litter across all years
- avgLitter: Average litter per survey
- litter_slope: Trend slope for linear regression
- litter_intercept: Y-intercept for predictions
- predicted_litter_2025: Predicted litter for 2025
- cluster: Beach grouping based on patterns
- robustGrowthRate: Growth rate of litter over time

Sample data (first 3 beaches):
{df.head(3).to_string()}
"""
        
        # System prompt with dataset context
        system_prompt = """You are Simek, an intelligent assistant for the ReefSpark platform. 
ReefSpark is dedicated to predicting reef bleaching events and providing unified, clean oceanographic data insights to help protect coral reefs.

You help users analyze marine data including beach litter patterns from European beaches (2012-2023), which serves as environmental indicators for reef health.

Your capabilities:
1. Answer questions about environmental trends, patterns, and statistics related to reef health
2. Compare data between beaches and countries to identify environmental stressors
3. Make predictions using the provided slope and intercept values to forecast future conditions
4. Explain data insights and trends that affect coral reef ecosystems
5. Calculate statistics and aggregations to support reef conservation efforts

Our mission: We provide unified, clean data and insights to predict and prevent reef bleaching, addressing the current lack of standardized oceanographic data.

When making predictions:
- Use the formula: predicted_value = litter_slope × year + litter_intercept
- For example, for 2026: predicted_2026 = litter_slope × 2026 + litter_intercept
- Explain your calculations clearly and relate findings to reef health when relevant

Be conversational, helpful, and data-driven. If you need to perform calculations, show your work.
"""
        
        # Call Google Gemini API (using free tier model)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        full_prompt = f"{system_prompt}\n\nDataset Context:\n{dataset_summary}\n\nUser Question: {user_message}"
        
        response = model.generate_content(full_prompt)
        assistant_message = response.text
        
        # Estimate tokens (Gemini doesn't provide exact count in basic API)
        estimated_tokens = len(full_prompt.split()) + len(assistant_message.split())
        
        return jsonify({
            'response': assistant_message,
            'model': 'gemini-2.5-flash',
            'tokens': estimated_tokens
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
