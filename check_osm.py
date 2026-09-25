"""
Check how well OpenStreetMap stand names match the challenge data
=================================================================

Run after osm_distances.py has finished, before wiring the distances into the
model. Prints one number that decides everything else: the percentage of
departures for which the stand-to-runway distance is known at all.

In this solution the match rate was 96 %, with Munich the weakest at 57 %
because OSM has sub-stands (111A, 111B) where the data has only 111. The main
pipeline handles that with a fallback on the base name.

Licence: GNU GPL v3.0 only.
"""

import os
import re

import numpy as np
import pandas as pd

# Set explicit paths here if needed; otherwise they are located automatically.
PATH_OSM = r''     # e.g. r'C:\path\to\cache\stand_dist.parquet'
PATH_TRAIN = r''   # e.g. r'C:\path\to\cache\train.parquet'


def locate(name, explicit):
    """Look for a file in the usual places: working folder, script folder,
    Downloads, Desktop."""
    if explicit:
        return explicit
    home = os.path.expanduser('~')
    places = [
        os.path.join('cache', name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', name),
        os.path.join(home, 'Downloads', 'cache', name),
        os.path.join(home, 'Desktop', 'cache', name),
        os.path.join(home, 'data', 'cache', name),
    ]
    for m in places:
        if os.path.exists(m):
            return m
    print(f'{name} not found. Looked in:')
    for m in places:
        print('   ', os.path.abspath(m))
    return None


def norm_stand(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = re.sub(r'[^A-Z0-9]', '', str(s).upper().strip())
    m = re.match(r'^([A-Z]*)0*([0-9]+)([A-Z]*)$', s)
    return f'{m.group(1)}{int(m.group(2))}{m.group(3)}' if m else s


def norm_rwy(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = re.sub(r'[^A-Z0-9]', '', str(s).upper().strip())
    m = re.match(r'^0*([0-9]{1,2})([LRC]?)$', s)
    return f'{int(m.group(1)):02d}{m.group(2)}' if m else s


def main():
    p_osm = locate('stand_dist.parquet', PATH_OSM)
    p_tr = locate('train.parquet', PATH_TRAIN)
    if not p_osm or not p_tr:
        raise SystemExit('\nSet PATH_OSM / PATH_TRAIN at the top of this file.')
    print(f'OSM file : {os.path.abspath(p_osm)}')
    print(f'data     : {os.path.abspath(p_tr)}\n')

    t = pd.read_parquet(p_osm)
    d = pd.read_parquet(p_tr)
    print(f'OSM: {len(t)} stand-runway pairs, {t.apt.nunique()} airports')
    print(f'data: {len(d)} departures\n')

    d = d[['apt', 'STAND_mvt', 'RUNWAY_mvt']].copy()
    d['stand_n'] = d.STAND_mvt.map(norm_stand)
    d['runway_n'] = d.RUNWAY_mvt.map(norm_rwy)
    m = d.merge(t, on=['apt', 'stand_n', 'runway_n'], how='left')

    print('=' * 58)
    print('SHARE OF FLIGHTS WITH A KNOWN DISTANCE')
    print('=' * 58)
    g = m.groupby('apt').agg(
        flights=('stand_n', 'size'),
        matched_pct=('dist_prava_m', lambda s: round(s.notna().mean() * 100, 1)),
        with_path_pct=('dist_taxi_m', lambda s: round(s.notna().mean() * 100, 1)),
        median_m=('dist_taxi_m', lambda s: round(s.median(), 0) if s.notna().any() else np.nan),
    )
    print(g.to_string())

    uk = m.dist_prava_m.notna().mean() * 100
    put = m.dist_taxi_m.notna().mean() * 100
    print(f'\nTOTAL: matched {uk:.1f} %, with a network path {put:.1f} %')
    if uk >= 70:
        print('-> good, the feature is worth using')
    elif uk >= 40:
        print('-> usable, but the name normalisation could be improved')
    else:
        print('-> too weak; compare the name samples printed below')

    q = m[m.dist_taxi_m.notna()].copy()
    if len(q):
        q['ratio'] = q.dist_taxi_m / q.dist_prava_m.replace(0, np.nan)
        print('\n' + '=' * 58)
        print('PATH QUALITY (network path over straight-line distance)')
        print('=' * 58)
        print(q.ratio.describe(percentiles=[.5, .9, .99]).round(2).to_string())
        print(f'share with a ratio above 3 (suspect paths): {(q.ratio > 3).mean() * 100:.1f} %')
        print('median distance per airport (m):')
        print(q.groupby('apt').dist_taxi_m.median().round(0).to_string())

    print('\n' + '=' * 58)
    print('NAME SAMPLES (first 25 per airport where matching is weak)')
    print('=' * 58)
    weak = g[g.matched_pct < 70].index.tolist() or list(g.index[:2])
    for apt in weak[:4]:
        a = sorted(set(t[t.apt == apt].stand_n.dropna()))[:25]
        b = sorted(set(d[d.apt == apt].stand_n.dropna()))[:25]
        print(f'\n{apt}')
        print(f'  OSM  : {a}')
        print(f'  data : {b}')


if __name__ == '__main__':
    main()
