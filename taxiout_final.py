"""
PRC Data Challenge 2026 - taxi-out time prediction
==================================================
Team joyous-rainbow. Official score 299.23 s (RMSE), rank 91 of 191.

The error in this task is not spread evenly. About 2.9 % of departures produce
71 % of the squared error, and they split into two groups with different causes,
so the pipeline treats them separately:

  main model     LightGBM with LINEAR LEAVES, for departures with a valid
                 Network Manager record. Linear leaves matter more here than any
                 feature: a constant-leaf tree cannot predict a value above the
                 largest one seen in training, which is exactly where the error
                 lives. Local RMSE 364.0 -> 338.5, official 333.1 -> 301.8.

  special model  three models combined, for departures with no Network Manager
                 record. All NM columns are missing at once for these rows, and
                 at some airports the airport system wrote the SCHEDULED
                 off-block time into the block-time field, so the recorded
                 taxi-out measures the delay of the flight rather than taxiing
                 (51 % of such rows at Rome, values of up to 24 hours).

Validation holds out January and July 2025 - the same months as the ranking set -
and trains on the remaining ten. A random K-fold would mix flights from the same
day and stand across the split and give an optimistic number that does not hold
on 2026 data.

Usage
-----
    pip install pandas numpy pyarrow lightgbm requests    (optuna: optional)
    python osm_distances.py      # optional, writes cache/stand_dist.parquet
    python taxiout_final.py      # weather, features, validation, submission file

Set DATA_DIR and TEAM_NAME below. Hyperparameters from a 100-trial Optuna search
are hard-coded as defaults, so the published result is reproducible without
repeating the search.

Licence: GNU GPL v3.0 only.
"""

import os
import re
import sys
import gc
import glob
import json
import time
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# SETTINGS
# ============================================================================
DATA_DIR  = r'data'            # folder with training_*.parquet and ranking.parquet
TEAM_NAME = 'joyous-rainbow'   # used for the submission file name
VERSION   = 1                  # increment for every submission
CACHE_DIR = 'cache'
FEATURE_VERSION = 5       # bump when features change; a stale cache is rebuilt
RUN_OPTUNA = False        # True re-runs the hyperparameter search (hours)
OPTUNA_TRIALS = 100
WEATHER_PASSES = 4        # retries for the rate-limited METAR service

N_WORKERS    = max(1, os.cpu_count() - 2)
VALID_MONTHS = [1, 7]                 # same months as the ranking set
CLIP_LO, CLIP_HI = 60, 10800          # training filter; 3600 measured 10 s worse
USE_WEATHER  = True                   # the pipeline runs without weather too

# time zones of the ten airports
AIRPORT_TZ = {
    'EDDF': 'Europe/Berlin',   'EDDM': 'Europe/Berlin',  'EGLL': 'Europe/London',
    'EHAM': 'Europe/Amsterdam','LEBL': 'Europe/Madrid',  'LEMD': 'Europe/Madrid',
    'LFPG': 'Europe/Paris',    'LIRF': 'Europe/Rome',    'LSZH': 'Europe/Zurich',
    'LTFM': 'Europe/Istanbul',
}
AIRPORT_LL = {   # latitude, longitude
    'EDDF': (50.033, 8.570),  'EDDM': (48.353, 11.786), 'EGLL': (51.470, -0.454),
    'EHAM': (52.309, 4.764),  'LEBL': (41.297, 2.078),  'LEMD': (40.472, -3.561),
    'LFPG': (49.010, 2.548),  'LIRF': (41.800, 12.239), 'LSZH': (47.458, 8.548),
    'LTFM': (41.262, 28.742),
}

RAW_COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt',
            'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt',
            'TAXITIME_SEC_mvt', 'LOBT_flt', 'MARKET_SEGMENT_flt', 'IOBT_flt',
            'FLIGHT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt',
            'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']

CATS = ['apt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
        'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_TYPE_flt']
NM_FEATS   = ['proxy', 'd_lobt', 'd_eobt', 'lobt_eobt', 'lobt_iobt', 'aobt_lobt',
              'plan_dur', 'flown_dur']
TIME_FEATS = ['d_sched', 'loc_hour', 'loc_dow', 'loc_minofday', 'doy', 'is_weekend',
              'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
              'traffic_bin']
OSM_FEATS  = ['dist_taxi_m', 'dist_prava_m', 'dist_odnos']
TI_FEATS   = ['ti_mean_1800', 'ti_mean_3600', 'ti_n_3600', 'ti_ratio_3600']
RG_FEATS   = ['rg_mean_3600', 'rg_long_3600', 'rg_mean_10800', 'rg_long_10800', 'rg_dev_3600']
CONG_FEATS = ['n_all_600', 'n_all_1800', 'n_all_3600', 'n_dep_600', 'n_dep_1800',
              'n_dep_3600', 'n_rwy_600', 'n_rwy_1800', 'n_rwy_3600', 'q_rwy',
              'rwy_share_3600']
WX_FEATS   = ['temp', 'prcp', 'snow', 'wspd', 'vis', 'is_freezing', 'is_precip',
              'day_tmin', 'day_snow', 'deice_risk']
TE_FEATS   = ['te_stand_rwy', 'te_stand', 'te_rwy']

MAIN_FEATS = (CATS + TIME_FEATS + NM_FEATS + CONG_FEATS + TI_FEATS + RG_FEATS
              + OSM_FEATS + TE_FEATS + WX_FEATS)
SPEC_FEATS = ['apt', 'STAND_mvt', 'RUNWAY_mvt', 'ADES_mvt', 'd_sched', 'loc_hour',
              'loc_dow', 'doy'] + CONG_FEATS + TI_FEATS

# Linear leaves let the model extrapolate above the largest value seen in
# training. Measured on January and July 2025: overall RMSE 364.0 -> 338.5,
# error on long flights 1,666 -> 1,181. Largest single gain in this solution.
LINEAR_TREE = True
PARAMS_MAIN = dict(objective='l2',
                   # found by a 100-trial Optuna search with the time-based split
                   learning_rate=0.0176, num_leaves=370, min_data_in_leaf=108,
                   feature_fraction=0.5106, bagging_fraction=0.9652, bagging_freq=1,
                   lambda_l1=0.0042, lambda_l2=4.034,
                   cat_smooth=94, max_cat_threshold=29,
                   verbose=-1, num_threads=N_WORKERS)
if LINEAR_TREE:
    # linear leaves need roughly half the leaves of a constant-leaf model
    PARAMS_MAIN.update(linear_tree=True, linear_lambda=1.0, num_leaves=185)
ROUNDS_MAIN = 4000        # early stopping decides; ~1,800 with these settings


# ============================================================================
# PART 0: DATA PREPARATION (one process per monthly file)
# ============================================================================
def _weather_meteostat(t0, t1):
    """Meteostat fallback; handles both the 1.x and 2.x API."""
    import meteostat as ms
    frames = []
    for apt, (lat, lon) in AIRPORT_LL.items():
        try:
            if hasattr(ms, 'TimeSeries'):                      # meteostat 2.x
                df = ms.hourly(ms.Point(lat, lon, 0), start=t0, end=t1).fetch()
            else:                                              # meteostat 1.x
                df = ms.Hourly(ms.Point(lat, lon), t0, t1).fetch()
            if df is None or df.empty:
                continue
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'time'})
            for c in ('temp', 'dwpt', 'prcp', 'snow', 'wspd', 'vis'):
                if c not in df:
                    df[c] = np.nan
            df = df[['time', 'temp', 'dwpt', 'prcp', 'snow', 'wspd', 'vis']]
            df['apt'] = apt
            frames.append(df)
            print(f'  meteostat {apt}: {len(df)} hours')
        except Exception as e:
            print(f'  meteostat {apt}: {e}')
    return pd.concat(frames, ignore_index=True) if frames else None


def _fetch_iem_station(apt, t0, t1, tries=6):
    """One airport from the Iowa Environmental Mesonet, retrying on HTTP 429.

    The service rate-limits requests, so the wait grows between attempts and each
    airport is cached as soon as it arrives, to avoid fetching it twice.
    """
    import io
    import urllib.error
    import urllib.request

    cache = f'{CACHE_DIR}/wx_{apt}.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    url = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
           'data=tmpf&data=dwpf&data=sknt&data=p01i&data=vsby&data=wxcodes'
           '&tz=UTC&format=onlycomma&missing=empty&trace=0.0001&latlon=no&report_type=3'
           f'&station={apt}&year1={t0.year}&month1={t0.month}&day1={t0.day}'
           f'&year2={t1.year}&month2={t1.month}&day2={t1.day}')

    wait = 15
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'prc-taxiout/1.0'})
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = r.read().decode()
            df = pd.read_csv(io.StringIO(raw))
            if df.empty:
                return None
            wx = df.get('wxcodes', pd.Series('', index=df.index)).fillna('').astype(str)
            out = pd.DataFrame({
                'time': pd.to_datetime(df.valid, utc=True, errors='coerce'),
                'temp': (pd.to_numeric(df.tmpf, errors='coerce') - 32) * 5 / 9,
                'prcp': pd.to_numeric(df.p01i, errors='coerce') * 25.4,
                'wspd': pd.to_numeric(df.sknt, errors='coerce') * 1.852,
                'vis':  pd.to_numeric(df.vsby, errors='coerce') * 1.609,
                'snow': wx.str.contains('SN|FZ|PL|GS|GR', regex=True).astype(float),
            }).dropna(subset=['time'])
            out['apt'] = apt
            os.makedirs(CACHE_DIR, exist_ok=True)
            out.to_parquet(cache, index=False)
            print(f'  METAR {apt}: {len(out)} observations')
            return out
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries:
                print(f'  METAR {apt}: rate limited, waiting {wait}s '
                      f'(attempt {attempt}/{tries})')
                time.sleep(wait)
                wait = min(wait * 2, 240)
                continue
            print(f'  METAR {apt}: {e}')
            return None
        except Exception as e:
            if attempt < tries:
                time.sleep(wait)
                wait = min(wait * 2, 240)
                continue
            print(f'  METAR {apt}: {e}')
            return None
    return None


def _weather_iem(t0, t1):
    """METAR observations: actual readings at the airport, more accurate than a
    model for visibility and freezing conditions. Fetched one airport at a time,
    with a pause, because the service rate-limits requests.

    Of all weather variables only VISIBILITY measurably helps here (4.5 s).
    Temperature and precipitation act at the level of a day; visibility triggers
    low-visibility procedures that slow every ground movement in that period."""
    frames = []
    for i, apt in enumerate(AIRPORT_LL):
        df = _fetch_iem_station(apt, t0, t1)
        if df is not None and len(df):
            frames.append(df)
        if i < len(AIRPORT_LL) - 1:
            time.sleep(10)          # pause between airports
    if not frames:
        return None
    missing = set(AIRPORT_LL) - {f.apt.iloc[0] for f in frames}
    if missing:
        print(f'  missing airports: {sorted(missing)} '
              f'(run again; already fetched airports are skipped)')
    return pd.concat(frames, ignore_index=True)


def load_weather():
    """Hourly weather per airport: METAR first, meteostat as a fallback.
    Cached on disk, so it is fetched only once."""
    path = f'{CACHE_DIR}/weather.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not USE_WEATHER:
        return None

    t0, t1 = pd.Timestamp('2024-12-25'), pd.Timestamp('2026-08-01')
    wx = None
    for fn in (_weather_iem, _weather_meteostat):
        try:
            wx = fn(t0, t1)
        except ImportError as e:
            print(f'  skipping ({e})')
            continue
        except Exception as e:
            print(f'  source failed: {e}')
            continue
        if wx is not None and len(wx):
            break
    if wx is None or not len(wx):
        print('weather unavailable, continuing without it')
        return None

    wx['time'] = pd.to_datetime(wx.time, utc=True).dt.floor('h')
    wx = wx.groupby(['apt', 'time'], as_index=False).mean(numeric_only=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    wx.to_parquet(path, index=False)
    print(f'weather: {len(wx)} rows, {wx.apt.nunique()} airports')
    return wx


def add_congestion(df):
    """Movement counts in windows around MVT_TIME, per airport and per runway.

    NOTE: the timestamps are in MICROseconds, so the divisor is 10**6. Dividing
    by 10**9 silently turns a one-hour window into a seven-day one; that bug cost
    a full round of measurements before it was caught.
    """
    df = df.sort_values('ts', kind='mergesort').reset_index(drop=True)
    n = len(df)
    out = {f'{p}{w}': np.zeros(n, np.int32)
           for w in (600, 1800, 3600) for p in ('n_all_', 'n_dep_', 'n_rwy_')}
    out['q_rwy'] = np.zeros(n, np.int32)

    for _, idx in df.groupby('apt', observed=True).indices.items():
        t = df.ts.values[idx]
        is_dep = (df.PHASE_mvt.values[idx] == 'DEP').astype(np.int32)
        cdep = np.concatenate([[0], np.cumsum(is_dep)])
        for w in (600, 1800, 3600):
            lo, hi = np.searchsorted(t, t - w), np.searchsorted(t, t + w)
            out[f'n_all_{w}'][idx] = hi - lo
            out[f'n_dep_{w}'][idx] = cdep[hi] - cdep[lo]

        rwy = df.RUNWAY_mvt.values[idx]
        for r in pd.unique(rwy):
            m = np.where(rwy == r)[0]
            tr = t[m]
            for w in (600, 1800, 3600):
                lo, hi = np.searchsorted(tr, tr - w), np.searchsorted(tr, tr + w)
                out[f'n_rwy_{w}'][idx[m]] = hi - lo
            cdr = np.concatenate([[0], np.cumsum(is_dep[m])])
            lo = np.searchsorted(tr, tr - 1200)
            out['q_rwy'][idx[m]] = cdr[np.arange(len(m))] - cdr[lo]

    for k, v in out.items():
        df[k] = v
    # share of the runway in the airport's traffic: a proxy for the configuration
    df['rwy_share_3600'] = (df.n_rwy_3600 / df.n_all_3600.clip(lower=1)).astype(np.float32)
    return df


def add_taxiin(df):
    """Mean taxi-in of ARRIVALS over the previous 30 and 60 minutes.

    A congestion measure that survives into the ranking set: only departure
    taxi-out was blanked, arrival taxi-in is still there. Strictly backward
    looking, never forward, so no information leaks from the future.
    """
    n = len(df)
    out = {c: np.full(n, np.nan, np.float32) for c in
           ('ti_mean_1800', 'ti_mean_3600', 'ti_ratio_3600')}
    out['ti_n_3600'] = np.zeros(n, np.int32)

    for _, idx in df.groupby('apt', observed=True).indices.items():
        t = df.ts.values[idx]
        tx = df.TAXITIME_SEC_mvt.values[idx].astype(float)
        arr = (df.PHASE_mvt.values[idx] == 'ARR') & np.isfinite(tx) & (tx > 0) & (tx < 7200)
        ta, xa = t[arr], tx[arr]
        if len(ta) < 10:
            continue
        cs = np.concatenate([[0], np.cumsum(xa)])
        cn = np.arange(len(ta) + 1)
        base = np.median(xa)
        for w in (1800, 3600):
            lo, hi = np.searchsorted(ta, t - w), np.searchsorted(ta, t)
            cnt, s = cn[hi] - cn[lo], cs[hi] - cs[lo]
            mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
            out[f'ti_mean_{w}'][idx] = mean
            if w == 3600:
                out['ti_n_3600'][idx] = cnt
                out['ti_ratio_3600'][idx] = mean / base
    for k, v in out.items():
        df[k] = v
    return df



def add_regime(df):
    """Airport regime from PAST departures: mean (take-off - AOBT_3) and the share
    of flights where that difference exceeds 30 min, over the last 1 and 3 hours.

    Available in the ranking set too, since only taxi-out was blanked. Captures
    multi-hour regimes such as de-icing or flow restrictions. The daily share of
    affected flights varies 5.8x more than chance would give, and the
    hour-to-hour correlation is 0.556 - the clustering is real. The measured gain
    is nevertheless small: the NM time difference already carries the result.
    """
    n = len(df)
    out = {c: np.full(n, np.nan, np.float32) for c in RG_FEATS}
    px = (df.MVT_TIME_UTC_mvt - df.AOBT_3_flt).dt.total_seconds().values

    for _, idx in df.groupby('apt', observed=True).indices.items():
        t = df.ts.values[idx]
        p = px[idx]
        ok = (df.PHASE_mvt.values[idx] == 'DEP') & np.isfinite(p) & (p > 0) & (p < 10800)
        to, xo = t[ok], p[ok]
        if len(to) < 20:
            continue
        cs = np.concatenate([[0], np.cumsum(xo)])
        cl = np.concatenate([[0], np.cumsum(xo > 1800)])
        cn = np.arange(len(to) + 1)
        base = np.median(xo)
        for w in (3600, 10800):
            lo, hi = np.searchsorted(to, t - w), np.searchsorted(to, t)
            cnt = cn[hi] - cn[lo]
            mean = np.where(cnt >= 3, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)
            out[f'rg_mean_{w}'][idx] = mean
            out[f'rg_long_{w}'][idx] = np.where(cnt >= 3, (cl[hi] - cl[lo]) / np.maximum(cnt, 1), np.nan)
            if w == 3600:
                out['rg_dev_3600'][idx] = mean / base
    for k, v in out.items():
        df[k] = v
    return df


def _norm_stand(s):
    """'A 12', 'A012', 'A12' -> 'A12'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = re.sub(r'[^A-Z0-9]', '', str(s).upper().strip())
    m = re.match(r'^([A-Z]*)0*([0-9]+)([A-Z]*)$', s)
    return f'{m.group(1)}{int(m.group(2))}{m.group(3)}' if m else s


def _norm_rwy(s):
    """'9L', '09 L' -> '09L'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = re.sub(r'[^A-Z0-9]', '', str(s).upper().strip())
    m = re.match(r'^0*([0-9]{1,2})([LRC]?)$', s)
    return f'{int(m.group(1)):02d}{m.group(2)}' if m else s


def _osm_table():
    """Stand-to-runway-threshold distances produced by osm_distances.py.

    Besides the exact name, a second table is built on the base name without the
    trailing letter, because OSM sometimes has sub-stands (111A, 111B) where the
    challenge data has only 111. Without it, Munich matches just 57 % of rows.
    """
    p = f'{CACHE_DIR}/stand_dist.parquet'
    if not os.path.exists(p):
        return None, None
    t = pd.read_parquet(p)
    t = t.dropna(subset=['dist_prava_m'])
    tocno = t.groupby(['apt', 'stand_n', 'runway_n'], as_index=False)[
        ['dist_taxi_m', 'dist_prava_m']].mean()
    baza = t.copy()
    baza['stand_n'] = baza.stand_n.str.replace(r'[A-Z]+$', '', regex=True)
    baza = baza[baza.stand_n != '']
    baza = baza.groupby(['apt', 'stand_n', 'runway_n'], as_index=False)[
        ['dist_taxi_m', 'dist_prava_m']].mean()
    return tocno, baza


def add_osm(d):
    """Join distances: exact stand name first, then the base name."""
    tocno, baza = _osm_table()
    if tocno is None:
        for c in OSM_FEATS:
            d[c] = np.nan
        return d

    d['_s'] = d.STAND_mvt.map(_norm_stand)
    d['_r'] = d.RUNWAY_mvt.map(_norm_rwy)
    keys = {'apt': 'apt', 'stand_n': '_s', 'runway_n': '_r'}
    for tab, suf in ((tocno, ''), (baza, '_b')):
        t = tab.rename(columns={'stand_n': '_s', 'runway_n': '_r',
                                'dist_taxi_m': f'dist_taxi_m{suf}',
                                'dist_prava_m': f'dist_prava_m{suf}'})
        d = d.merge(t, on=['apt', '_s', '_r'], how='left')
    d['dist_taxi_m'] = d.dist_taxi_m.fillna(d.dist_taxi_m_b)
    d['dist_prava_m'] = d.dist_prava_m.fillna(d.dist_prava_m_b)
    d['dist_odnos'] = d.dist_taxi_m / d.dist_prava_m.replace(0, np.nan)
    # detours caused by gaps in the OSM network: keep the straight line, drop the path
    lose = d.dist_odnos > 3
    d.loc[lose, ['dist_taxi_m', 'dist_odnos']] = np.nan
    for c in OSM_FEATS:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(np.float32)
    return d.drop(columns=['_s', '_r', 'dist_taxi_m_b', 'dist_prava_m_b'])


def add_local_time(d):
    """Local time with cyclical encoding; taxi-out follows local, not UTC, rhythm."""
    loc = pd.Series(pd.NaT, index=d.index, dtype='datetime64[ns]')
    for apt, tz in AIRPORT_TZ.items():
        m = (d.apt == apt).values
        if m.any():
            loc[m] = d.loc[m, 'MVT_TIME_UTC_mvt'].dt.tz_convert(tz).dt.tz_localize(None)
    lt = loc.dt
    d['loc_hour']     = lt.hour.fillna(12).astype(np.int8)
    d['loc_dow']      = lt.dayofweek.fillna(0).astype(np.int8)
    d['loc_minofday'] = (lt.hour * 60 + lt.minute).fillna(720).astype(np.int16)
    d['doy']          = d.MVT_TIME_UTC_mvt.dt.dayofyear.astype(np.int16)
    d['month']        = d.MVT_TIME_UTC_mvt.dt.month.astype(np.int8)
    d['is_weekend']   = (d.loc_dow > 4).astype(np.int8)
    h, w, y = d.loc_hour, d.loc_dow, d.doy
    d['hour_sin'], d['hour_cos'] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    d['dow_sin'],  d['dow_cos']  = np.sin(2*np.pi*w/7),  np.cos(2*np.pi*w/7)
    d['doy_sin'],  d['doy_cos']  = np.sin(2*np.pi*y/365), np.cos(2*np.pi*y/365)
    # morning peak / evening peak / night / other
    d['traffic_bin'] = np.select(
        [h < 6, (h >= 6) & (h < 10), (h >= 17) & (h < 21)], [0, 2, 3], default=1
    ).astype(np.int8)
    return d


def add_weather(d, wx):
    """Join weather on the hour of the movement and derive de-icing indicators.

    day_tmin    lowest temperature that day at that airport
    day_snow    whether snow or freezing precipitation occurred that day
    deice_risk  cold combined with precipitation
    """
    if wx is None:
        for c in WX_FEATS:
            d[c] = np.nan
        return d

    d['_h'] = d.MVT_TIME_UTC_mvt.dt.floor('h')
    cols = ['apt', 'time'] + [c for c in ('temp', 'prcp', 'snow', 'wspd', 'vis') if c in wx]
    d = d.merge(wx[cols].rename(columns={'time': '_h'}), on=['apt', '_h'], how='left')
    for c in ('temp', 'prcp', 'snow', 'wspd', 'vis'):
        if c not in d:
            d[c] = np.nan

    # daily indicators: a regime lasts hours, so the day says more than the hour
    day = d.assign(_d=d._h.dt.date).groupby(['apt', '_d']).agg(
        day_tmin=('temp', 'min'), day_snow=('snow', 'max')).reset_index()
    d['_d'] = d._h.dt.date
    d = d.merge(day, on=['apt', '_d'], how='left')

    d['is_freezing'] = (d.temp < 3).astype(np.float32)
    d['is_precip'] = ((d.prcp.fillna(0) > 0) | (d.snow.fillna(0) > 0)).astype(np.float32)
    d['deice_risk'] = (d.is_freezing * (1 + d.is_precip)
                       + (d.day_tmin < 0).astype(float) + d.day_snow.fillna(0)).astype(np.float32)
    for c in WX_FEATS:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(np.float32)
    return d.drop(columns=['_h', '_d'])


def build_features(path, wx=None):
    """One monthly file -> a table of departures with every feature."""
    df = pd.read_parquet(path, columns=RAW_COLS)
    df['apt'] = np.where(df.PHASE_mvt == 'DEP', df.ADEP_mvt, df.ADES_mvt)
    df['ts'] = (df.MVT_TIME_UTC_mvt.astype('int64') // 10**6).astype(np.int64)
    df = add_congestion(df)
    df = add_taxiin(df)
    df = add_regime(df)

    d = df[df.PHASE_mvt == 'DEP'].copy()
    del df
    gc.collect()

    d['proxy']     = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    d['d_sched']   = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['d_lobt']    = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    d['d_eobt']    = (d.MVT_TIME_UTC_mvt - d.EOBT_1_flt).dt.total_seconds()
    d['lobt_eobt'] = (d.LOBT_flt - d.EOBT_1_flt).dt.total_seconds()
    d['lobt_iobt'] = (d.LOBT_flt - d.IOBT_flt).dt.total_seconds()
    d['aobt_lobt'] = (d.AOBT_3_flt - d.LOBT_flt).dt.total_seconds()
    d['plan_dur']  = (d.ARVT_1_flt - d.EOBT_1_flt).dt.total_seconds()
    d['flown_dur'] = (d.ARVT_3_flt - d.AOBT_3_flt).dt.total_seconds()
    d['has_nm']    = d.AOBT_3_flt.notna()

    d = add_local_time(d)
    d = add_osm(d)
    d = add_weather(d, wx)

    keep = ['MVT_ID_mvt', 'apt', 'month', 'has_nm', 'TAXITIME_SEC_mvt'] \
        + CATS[1:] + TIME_FEATS + NM_FEATS + CONG_FEATS + TI_FEATS + RG_FEATS \
        + OSM_FEATS + WX_FEATS
    keep = [c for c in dict.fromkeys(keep) if c in d.columns]
    for c in keep:
        # MVT_ID_mvt must NOT be downcast to float32: the identifiers have nine
        # digits and float32 holds about seven significant ones, so distinct
        # flights would collapse onto the same id and the submission file would
        # be silently invalid (344,841 unique ids became 22,048 when this hit).
        if c == 'MVT_ID_mvt':
            d[c] = d[c].astype('int64')
        elif d[c].dtype == 'float64':
            d[c] = d[c].astype(np.float32)
    return d[keep]


def _worker(path, wx):
    t0 = time.time()
    out = build_features(path, wx)
    name = os.path.basename(path).replace('.parquet', '')
    out.to_parquet(f'{CACHE_DIR}/feat_{name}.parquet', index=False)
    print(f'  {name}: {len(out)} rows in {time.time()-t0:.0f}s', flush=True)
    return len(out)


def prepare(force=False):
    """Parallel preparation: one process per monthly file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tr_cache, rk_cache = f'{CACHE_DIR}/train.parquet', f'{CACHE_DIR}/rank.parquet'

    # The feature cache is valid only for the same code version. An older cache
    # is removed automatically, so that features without weather, or with broken
    # identifiers, are never used silently. The weather cache (wx_*.parquet) is
    # kept, since fetching it again is slow and rate-limited.
    vfile = f'{CACHE_DIR}/feature_version.txt'
    have = open(vfile).read().strip() if os.path.exists(vfile) else ''
    if have != str(FEATURE_VERSION):
        stale = [f for f in os.listdir(CACHE_DIR)
                 if f.startswith('feat_') or f in ('train.parquet', 'rank.parquet')]
        if stale:
            print(f'cache is from an older version ({have or "unknown"} -> {FEATURE_VERSION}), '
                  f'removing {len(stale)} files and recomputing')
            for f in stale:
                os.remove(f'{CACHE_DIR}/{f}')
        force = True

    if not force and os.path.exists(tr_cache) and os.path.exists(rk_cache):
        tr_c, rk_c = pd.read_parquet(tr_cache), pd.read_parquet(rk_cache)
        if rk_c.MVT_ID_mvt.nunique() == len(rk_c):
            return tr_c, rk_c
        print('cache has broken identifiers, recomputing')

    print('weather...')
    wx = load_weather()

    files = sorted(glob.glob(f'{DATA_DIR}/training_*.parquet'))
    if not files:
        raise SystemExit(
            f'\nNo input files in "{os.path.abspath(DATA_DIR)}".\n'
            f'Set DATA_DIR at the top of this file to the folder holding\n'
            f'training_*.parquet, ranking.parquet and submitting.parquet.')
    print(f'preparing {len(files)} months on {N_WORKERS} processes')
    t0 = time.time()
    with Pool(processes=N_WORKERS) as pool:
        pool.map(partial(_worker, wx=wx), files, chunksize=1)
    print(f'done in {time.time()-t0:.0f}s')

    parts = [pd.read_parquet(f'{CACHE_DIR}/feat_{os.path.basename(f)[:-8]}.parquet')
             for f in files]
    train = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    rank = build_features(f'{DATA_DIR}/ranking.parquet', wx)

    train.to_parquet(tr_cache, index=False)
    rank.to_parquet(rk_cache, index=False)
    open(vfile, 'w').write(str(FEATURE_VERSION))
    print('training', train.shape, ' ranking', rank.shape)
    return train, rank


# ============================================================================
# TARGET ENCODING AND CATEGORIES
# ============================================================================
def add_target_encoding(source, targets, smooth=20):
    """Statistics are computed ONLY on the training part (source), never on the
    validation months or the ranking set."""
    base = source[source.TAXITIME_SEC_mvt.between(CLIP_LO, CLIP_HI)]
    prior = base.TAXITIME_SEC_mvt.median()

    def make(keys, name):
        g = base.groupby(keys, observed=True).TAXITIME_SEC_mvt.agg(['median', 'size'])
        g[name] = (g['median'] * g['size'] + prior * smooth) / (g['size'] + smooth)
        return g[[name]].reset_index(), keys

    maps = [make(['apt', 'STAND_mvt', 'RUNWAY_mvt'], 'te_stand_rwy'),
            make(['apt', 'STAND_mvt'], 'te_stand'),
            make(['apt', 'RUNWAY_mvt'], 'te_rwy')]
    out = []
    for df in targets:
        df = df.copy()
        for m, keys in maps:
            df = df.merge(m, on=keys, how='left')
        for c in TE_FEATS:
            df[c] = df[c].fillna(prior).astype(np.float32)
        out.append(df)
    return out


def align_categories(train, others):
    train = train.copy()
    for c in CATS:
        train[c] = train[c].astype('category')
    res = []
    for df in others:
        df = df.copy()
        for c in CATS:
            df[c] = pd.Categorical(df[c].astype(str), categories=train[c].cat.categories)
        res.append(df)
    return train, res


def split_time(train):
    """Time-based split. This is where it differs from KFold(shuffle=True)."""
    m = train.month.isin(VALID_MONTHS)
    tr, va = train[~m].copy(), train[m].copy()
    tr, (va,) = align_categories(tr, [va])
    tr, va = add_target_encoding(tr, [tr, va])
    return tr, va


# ============================================================================
# PARTS 1 AND 2: MODELS
# ============================================================================
def train_main(tr, va=None, params=None, rounds=ROUNDS_MAIN):
    p = dict(PARAMS_MAIN)
    if params:
        p.update(params)
        if LINEAR_TREE:
            # the Optuna parameters were found without linear leaves; with them
            # the trees must be half the size or the model overfits
            p.update(linear_tree=True, linear_lambda=1.0,
                     num_leaves=max(31, int(p.get('num_leaves', 128) / 2)))
    m = tr.TAXITIME_SEC_mvt.between(CLIP_LO, CLIP_HI) & tr.has_nm
    ds = lgb.Dataset(tr.loc[m, MAIN_FEATS], tr.loc[m, 'TAXITIME_SEC_mvt'])
    valid, cb = [], []
    if va is not None:
        valid = [lgb.Dataset(va[MAIN_FEATS], va.TAXITIME_SEC_mvt, reference=ds)]
        cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)]
    return lgb.train(p, ds, num_boost_round=rounds, valid_sets=valid, callbacks=cb)


def fit_special(tr):
    """Group A: rows with no AOBT_3, where every NM timestamp is missing.

    At some airports the SCHEDULED off-block time was written into BLOCK_TIME
    (51 % of such rows at Rome), so taxi-out there measures the delay of the
    flight rather than taxiing. Three models over the same subgroup, combined:

      curve      cubic polynomial + hinge at three hours, per airport; the only
                 component that extrapolates arbitrarily far
      trees      LightGBM with linear leaves
      mixture    classifier p = P(taxi == delay), then
                 p * delay + (1 - p) * model for the remaining rows

    Measured on January and July 2025 (RMSE within the group): curve 1,908,
    trees 1,924, average of the two 1,863, mixture 1,860, all three 1,822. The
    classifier reaches AUC 0.956; its probability is shrunk by 0.8, because
    uncalibrated confidence is expensive when it is wrong.

    A robust (Huber) linear fit was also tried and is much worse (2,178): the
    extremes in this group are correct data, not noise, and must not be damped.
    """
    sub = tr[~tr.has_nm & tr.d_sched.notna()]
    models, KNOT = {}, 10800
    for apt, g in sub.groupby('apt', observed=True):
        if len(g) < 30:
            continue
        x, yy = g.d_sched.values, g.TAXITIME_SEC_mvt.values
        cap = float(np.quantile(yy, 0.999))
        if len(g) >= 150:
            b3 = np.polyfit(x, yy, 3)
            X = np.c_[x, np.maximum(x - KNOT, 0), np.ones(len(x))]
            bp = np.linalg.lstsq(X, yy, rcond=None)[0]
            models[apt] = ('blend', b3, bp, cap)
        else:
            models[apt] = ('lin', np.polyfit(x, yy, 1), None, cap)

    booster = clf = normal = None
    if len(sub) >= 2000:
        pr = dict(objective='l2', learning_rate=0.05, num_leaves=24,
                  min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, verbose=-1, num_threads=N_WORKERS,
                  linear_tree=True, linear_lambda=1.0)
        booster = lgb.train(pr, lgb.Dataset(sub[SPEC_FEATS], sub.TAXITIME_SEC_mvt),
                            num_boost_round=400)
        fb = ((sub.d_sched - sub.TAXITIME_SEC_mvt).abs() < 60).astype(int)
        if 0.005 < fb.mean() < 0.9:
            pc = dict(pr); pc.update(objective='binary')
            pc.pop('linear_tree'); pc.pop('linear_lambda')
            clf = lgb.train(pc, lgb.Dataset(sub[SPEC_FEATS], fb), num_boost_round=300)
            nn = sub[fb == 0]
            if len(nn) >= 500:
                normal = lgb.train(pr, lgb.Dataset(nn[SPEC_FEATS], nn.TAXITIME_SEC_mvt),
                                   num_boost_round=400)

    fallback = float(sub.TAXITIME_SEC_mvt.median()) if len(sub) else float(tr.TAXITIME_SEC_mvt.median())
    print(f'special model: {len(sub)} rows, {len(models)} airports'
          + (', + trees' if booster is not None else '')
          + (', + classifier' if normal is not None else ''))
    return models, fallback, KNOT, booster, clf, normal


def _special_curve(kind, b3, bp, x, knot):
    if kind == 'lin':
        return np.polyval(b3, x)
    piece = bp[0] * x + bp[1] * np.maximum(x - knot, 0) + bp[2]
    return 0.5 * np.polyval(b3, x) + 0.5 * piece


def predict_all(df, main_model, special):
    """Main model everywhere, then the three-model combination over group A."""
    models, fallback, knot, booster, clf, normal = special
    p = main_model.predict(df[MAIN_FEATS])
    apt = df.apt.astype(str).values
    miss = (~df.has_nm).values
    ds = df.d_sched.values
    if not miss.any():
        return np.clip(p, CLIP_LO, None)

    X = df.loc[miss, SPEC_FEATS]
    p_gbm = np.clip(booster.predict(X), CLIP_LO, None) if booster is not None else None
    p_mix = None
    if clf is not None and normal is not None:
        p_fb = 0.8 * clf.predict(X)                 # shrunk probability
        p_nrm = np.clip(normal.predict(X), CLIP_LO, None)
        p_mix = np.clip(p_fb * ds[miss] + (1 - p_fb) * p_nrm, CLIP_LO, None)

    idx_miss = np.where(miss)[0]
    for a, (kind, b3, bp, cap) in models.items():
        m = miss & (apt == str(a))
        if not m.any():
            continue
        pos = np.searchsorted(idx_miss, np.where(m)[0])
        kriva = np.clip(_special_curve(kind, b3, bp, ds[m], knot), CLIP_LO, cap)
        if p_gbm is None:
            val = kriva
        elif p_mix is None:
            val = 0.5 * kriva + 0.5 * p_gbm[pos]
        else:
            val = 0.5 * (0.5 * kriva + 0.5 * p_gbm[pos]) + 0.5 * p_mix[pos]
        p[m] = np.clip(val, CLIP_LO, cap)
    orphan = miss & ~np.isin(apt, [str(a) for a in models])
    p[orphan] = fallback
    p[miss & ~np.isfinite(ds)] = fallback
    return np.clip(p, CLIP_LO, None)


# ============================================================================
# EVALUATION
# ============================================================================
def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def report(va, p):
    """Report by group, because the error does not behave the same everywhere.

    A - no AOBT_3 (never matched to a Network Manager record)
    B - long flights with a valid AOBT_3 (NM logs pushback ~25 min late)
    C - regular flights
    """
    y = va.TAXITIME_SEC_mvt.values
    a = (~va.has_nm).values
    b = (~a) & (y > 2400)
    c = (~a) & (~b)

    print('\n=================== RESULT ====================')
    print(f'RMSE overall              : {rmse(p, y):8.1f}   <- leaderboard metric')
    for mth, nm in [(1, 'January'), (7, 'July')]:
        m = (va.month == mth).values
        if m.sum():
            print(f'RMSE {nm:<21}: {rmse(p[m], y[m]):8.1f}')
    print(f'MAE  overall              : {np.abs(p - y).mean():8.1f}')
    print('-- by group --')
    for nm, m in [('A no AOBT_3', a), ('B long, valid AOBT_3', b), ('C regular', c)]:
        if m.sum():
            print(f'{nm:<18} n={m.sum():7d}  RMSE {rmse(p[m], y[m]):8.1f}  '
                  f'MAE {np.abs(p[m] - y[m]).mean():7.1f}  share of error '
                  f'{((p[m]-y[m])**2).sum()/((p-y)**2).sum()*100:5.1f}%')
    e2 = (p - y) ** 2
    share = va.assign(e2=e2).groupby('apt', observed=True).e2.sum() / e2.sum() * 100
    print('-- share of error by airport (%) --')
    print(share.sort_values(ascending=False).round(1).to_string())
    print('================================================\n')


# ============================================================================
# PART 3: OPTUNA (time-based split, not KFold)
# ============================================================================
def run_optuna(train, n_trials=30, timeout_per_trial=2400):
    import optuna

    tr, va = split_time(train)
    special = fit_special(tr)
    yva = va.TAXITIME_SEC_mvt.values

    class Timeout:
        def __init__(self, sec):
            self.sec, self.t0 = sec, None

        def __call__(self, env):
            if self.t0 is None:
                self.t0 = time.time()
            if time.time() - self.t0 > self.sec:
                raise optuna.TrialPruned()

    def objective(trial):
        params = {
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 64, 512),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'lambda_l1':        trial.suggest_float('lambda_l1', 1e-3, 10, log=True),
            'lambda_l2':        trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
            'cat_smooth':       trial.suggest_int('cat_smooth', 5, 100),
            'max_cat_threshold': trial.suggest_int('max_cat_threshold', 16, 128),
        }
        p = dict(PARAMS_MAIN)
        p.update(params)
        m = tr.TAXITIME_SEC_mvt.between(CLIP_LO, CLIP_HI)
        ds = lgb.Dataset(tr.loc[m, MAIN_FEATS], tr.loc[m, 'TAXITIME_SEC_mvt'])
        dv = lgb.Dataset(va[MAIN_FEATS], yva, reference=ds)
        model = lgb.train(p, ds, num_boost_round=ROUNDS_MAIN, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     Timeout(timeout_per_trial)])
        pred = predict_all(va, model, special)
        trial.set_user_attr('best_iteration', model.best_iteration)
        trial.set_user_attr('rmse_normal', rmse(pred[yva <= 2400], yva[yva <= 2400]))
        return rmse(pred, yva)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=42),
        study_name='taxiout_lgbm', storage='sqlite:///optuna_taxiout.db',
        load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print('best RMSE:', round(study.best_value, 2))
    print('best parameters:', json.dumps(study.best_params, indent=1))
    json.dump({'params': study.best_params,
               'best_iteration': study.best_trial.user_attrs.get('best_iteration')},
              open(f'{CACHE_DIR}/best_params.json', 'w'), indent=1)
    return study.best_params


# ============================================================================
# MAIN FLOW
# ============================================================================
def run_validate(train, params=None):
    tr, va = split_time(train)
    print(f'training {len(tr)}, validation {len(va)}')
    model = train_main(tr, va, params)
    special = fit_special(tr)
    p = predict_all(va, model, special)
    report(va, p)
    print('main model only  :', round(rmse(np.clip(model.predict(va[MAIN_FEATS]), 0, None),
                                           va.TAXITIME_SEC_mvt), 1))
    print('with special model:', round(rmse(p, va.TAXITIME_SEC_mvt), 1))
    return model.best_iteration or ROUNDS_MAIN


def run_submit(train, rank, best_iter, params=None):
    tr, (rank,) = align_categories(train, [rank])
    tr, rank = add_target_encoding(tr, [tr, rank])
    model = train_main(tr, None, params, rounds=int(best_iter * 1.1))
    special = fit_special(tr)
    p = predict_all(rank, model, special)

    sub = pd.read_parquet(f'{DATA_DIR}/submitting.parquet')
    id_dtype = sub.MVT_ID_mvt.dtype
    out = pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.astype('int64').values, '_p': p})
    sub['_id64'] = sub.MVT_ID_mvt.astype('int64')
    sub = sub.drop(columns=['TAXITIME_SEC_mvt']).merge(
        out.rename(columns={'MVT_ID_mvt': '_id64'}), on='_id64', how='left')
    sub = sub.drop(columns=['_id64']).rename(columns={'_p': 'TAXITIME_SEC_mvt'})
    sub['MVT_ID_mvt'] = sub.MVT_ID_mvt.astype(id_dtype)
    n_exp = len(pd.read_parquet(f'{DATA_DIR}/submitting.parquet'))
    assert len(sub) == n_exp, f'file has {len(sub)} rows instead of {n_exp}'
    if sub.TAXITIME_SEC_mvt.isna().any():
        n = int(sub.TAXITIME_SEC_mvt.isna().sum())
        print(f'WARNING: {n} rows without a prediction, filling with the median')
        sub['TAXITIME_SEC_mvt'] = sub.TAXITIME_SEC_mvt.fillna(tr.TAXITIME_SEC_mvt.median())

    # File name per the challenge rules: <team-name>_v<n>.parquet
    name = f'{TEAM_NAME}_v{VERSION}.parquet'

    # The same checks the ranking script performs: identical rows, identical
    # identifiers, nothing missing and nothing extra.
    tpl = pd.read_parquet(f'{DATA_DIR}/submitting.parquet')
    assert len(sub) == len(tpl), f'{len(sub)} rows, expected {len(tpl)}'
    assert (sub.MVT_ID_mvt.astype('int64').values
            == tpl.MVT_ID_mvt.astype('int64').values).all(), 'identifiers do not match'
    assert list(sub.columns) == ['MVT_ID_mvt', 'TAXITIME_SEC_mvt'], 'wrong columns'
    assert sub.TAXITIME_SEC_mvt.notna().all(), 'missing predictions'
    assert np.isfinite(sub.TAXITIME_SEC_mvt).all(), 'infinite values'
    print('checks passed: row count, identifiers, columns, missing values')

    sub = sub[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    sub.to_parquet(name, index=False)
    print(f'\nsaved: {name}  ({len(sub)} rows)')
    print(sub.TAXITIME_SEC_mvt.describe().round(1).to_string())
    print(f'\nupload with:  mc cp {name} <alias>/<your-bucket>/{name}')
    print(f'next submission: set VERSION to {VERSION + 1}')


def collect_weather():
    """Several passes over METAR until all airports are collected.

    The service rate-limits requests (HTTP 429), so airports already fetched are
    cached and skipped on the next pass.
    """
    if not USE_WEATHER:
        return None
    for i in range(1, WEATHER_PASSES + 1):
        have = {f[3:7] for f in os.listdir(CACHE_DIR)} if os.path.exists(CACHE_DIR) else set()
        have = {a for a in AIRPORT_LL if a in have}
        if len(have) == len(AIRPORT_LL):
            break
        print(f'\n--- weather, pass {i}/{WEATHER_PASSES} '
              f'(have {len(have)}/{len(AIRPORT_LL)} airports) ---')
        if os.path.exists(f'{CACHE_DIR}/weather.parquet'):
            os.remove(f'{CACHE_DIR}/weather.parquet')
        try:
            load_weather()
        except Exception as e:
            print('  weather pass failed:', e)
        if i < WEATHER_PASSES:
            time.sleep(30)
    wx = load_weather()
    if wx is not None:
        print(f'weather ready: {wx.apt.nunique()}/{len(AIRPORT_LL)} airports, {len(wx)} rows')
    return wx


def main():
    t_start = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('=' * 60)
    print('STEP 1/4  weather (METAR)')
    print('=' * 60)
    collect_weather()

    print('\n' + '=' * 60)
    print('STEP 2/4  features')
    print('=' * 60)
    train, rank = prepare()

    params = None
    if RUN_OPTUNA:
        print('\n' + '=' * 60)
        print('STEP 2b   Optuna')
        print('=' * 60)
        params = run_optuna(train, OPTUNA_TRIALS)
    elif os.path.exists(f'{CACHE_DIR}/best_params.json'):
        params = json.load(open(f'{CACHE_DIR}/best_params.json'))['params']
        print('using hyperparameters from a previous Optuna search')

    print('\n' + '=' * 60)
    print('STEP 3/4  validation on January and July 2025')
    print('=' * 60)
    best_iter = run_validate(train, params)

    print('\n' + '=' * 60)
    print('STEP 4/4  training on all 12 months, writing the submission')
    print('=' * 60)
    run_submit(train, rank, best_iter, params)

    print(f'\ndone in {(time.time() - t_start) / 60:.1f} minutes')


if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f'Folder "{DATA_DIR}" does not exist. Set DATA_DIR at the top.')
    main()
