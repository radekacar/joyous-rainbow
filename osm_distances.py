"""
Stand-to-runway-threshold distances from OpenStreetMap
======================================================

Taxi-out time depends first of all on how far the aircraft physically has to
travel. Target encoding approximates that with an average per stand-runway pair,
but knows no geometry and can say nothing about a stand or a configuration it has
not seen.

This script:
  1. fetches the layout of each of the ten airports from the Overpass API
     (parking positions, taxiways, runways),
  2. builds the taxiway network as a graph,
  3. computes the shortest path from every stand to every runway threshold,
  4. writes cache/stand_dist.parquet

Takes 5 to 15 minutes, mostly waiting on Overpass. The result is cached.

Measured effect: 96 % of departures matched, median path/straight-line ratio 1.30,
local RMSE 361.3 -> 361.0, official 333.1 -> 333.7. Below the noise floor of this
problem; kept because it generalises to unseen stands, but it earns little.

Requires: pandas numpy pyarrow requests
Licence: GNU GPL v3.0 only.
"""

import heapq
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import requests

CACHE_DIR = 'cache'
OUT = f'{CACHE_DIR}/stand_dist.parquet'
RAW = f'{CACHE_DIR}/osm'                     # raw responses, so nothing is fetched twice

OVERPASS = ['https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter']

# approximate airport centres; the radius covers even the largest aerodromes
AIRPORTS = {
    'EDDF': (50.033, 8.570), 'EDDM': (48.353, 11.786), 'EGLL': (51.470, -0.454),
    'EHAM': (52.309, 4.764), 'LEBL': (41.297, 2.078),  'LEMD': (40.472, -3.561),
    'LFPG': (49.010, 2.548), 'LIRF': (41.800, 12.239), 'LSZH': (47.458, 8.548),
    'LTFM': (41.262, 28.742),
}
RADIUS_M = 6000


# ---------------------------------------------------------------- preuzimanje
def overpass_query(apt, lat, lon):
    """Parking positions, taxiways, taxilanes and runways around the airport."""
    return f"""
[out:json][timeout:180];
(
  node(around:{RADIUS_M},{lat},{lon})["aeroway"="parking_position"];
  way(around:{RADIUS_M},{lat},{lon})["aeroway"="parking_position"];
  way(around:{RADIUS_M},{lat},{lon})["aeroway"="taxiway"];
  way(around:{RADIUS_M},{lat},{lon})["aeroway"="taxilane"];
  way(around:{RADIUS_M},{lat},{lon})["aeroway"="runway"];
);
out body geom;
"""


def fetch(apt):
    os.makedirs(RAW, exist_ok=True)
    path = f'{RAW}/{apt}.json'
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))

    lat, lon = AIRPORTS[apt]
    q = overpass_query(apt, lat, lon)
    for url in OVERPASS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={'data': q}, timeout=240,
                                  headers={'User-Agent': 'prc-taxiout/1.0'})
                if r.status_code == 200:
                    data = r.json()
                    json.dump(data, open(path, 'w', encoding='utf-8'))
                    print(f'  {apt}: {len(data.get("elements", []))} elements')
                    return data
                print(f'  {apt}: status {r.status_code}, waiting')
            except Exception as e:
                print(f'  {apt}: {e}')
            time.sleep(20 * (attempt + 1))
    print(f'  {apt}: failed')
    return None


# ---------------------------------------------------------------- geometrija
def meters(lat1, lon1, lat2, lon2):
    """Distance in metres; a planar approximation is accurate enough over 10 km."""
    k = 111320.0
    dy = (lat2 - lat1) * k
    dx = (lon2 - lon1) * k * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def bearing(lat1, lon1, lat2, lon2):
    """Bearing from the first point to the second, in degrees (0 = north)."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.cos(math.radians(lon2 - lon1)))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def norm_stand(s):
    """Stand names are written inconsistently ('A12', 'A 12', 'A012'); reduce them
    to LETTERS+NUMBER without leading zeros."""
    if s is None:
        return None
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s)
    m = re.match(r'^([A-Z]*)0*([0-9]+)([A-Z]*)$', s)
    return f'{m.group(1)}{int(m.group(2))}{m.group(3)}' if m else s


def norm_rwy(s):
    """'09L', '9L', '09 L' -> '09L'."""
    if s is None:
        return None
    s = re.sub(r'[^A-Z0-9]', '', str(s).upper().strip())
    m = re.match(r'^0*([0-9]{1,2})([LRC]?)$', s)
    return f'{int(m.group(1)):02d}{m.group(2)}' if m else s


# ---------------------------------------------------------------- graf
class Graph:
    """The taxiway network. Nodes are coordinates rounded to about one metre,
    which joins the ends of adjacent segments that OSM does not share a node
    between - without this the graph is full of breaks."""

    def __init__(self, prec=5):
        self.prec = prec
        self.adj = {}

    def key(self, lat, lon):
        return (round(lat, self.prec), round(lon, self.prec))

    def add_way(self, geom, weight=1.0):
        for a, b in zip(geom[:-1], geom[1:]):
            ka, kb = self.key(a['lat'], a['lon']), self.key(b['lat'], b['lon'])
            if ka == kb:
                continue
            d = meters(a['lat'], a['lon'], b['lat'], b['lon']) * weight
            self.adj.setdefault(ka, []).append((kb, d))
            self.adj.setdefault(kb, []).append((ka, d))

    def nearest(self, lat, lon, max_m=400):
        """Nearest node in the network; used to attach stands and thresholds."""
        if not self.adj:
            return None
        best, bd = None, max_m
        for k in self.adj:
            d = meters(lat, lon, k[0], k[1])
            if d < bd:
                best, bd = k, d
        return best

    def dijkstra(self, src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist


# ---------------------------------------------------------------- obrada
def parse_airport(apt, data):
    """Returns (graph, stands, runway thresholds)."""
    g = Graph()
    stands, thresholds = {}, {}

    for el in data.get('elements', []):
        tags = el.get('tags', {})
        aero = tags.get('aeroway')
        geom = el.get('geometry')

        if aero in ('taxiway', 'taxilane') and geom:
            g.add_way(geom)

        elif aero == 'runway' and geom and len(geom) >= 2:
            # runways enter the graph at a higher cost: aircraft avoid taxiing on them
            g.add_way(geom, weight=1.5)
            ref = tags.get('ref') or tags.get('name') or ''
            a, b = geom[0], geom[-1]
            brg = bearing(a['lat'], a['lon'], b['lat'], b['lon'])
            for ident in [x for x in re.split(r'[/;]', ref) if x.strip()]:
                r = norm_rwy(ident)
                if not r or not r[:2].isdigit():
                    continue
                want = int(r[:2]) * 10
                # the threshold is the end from which take-off follows the designator
                diff_a = abs((brg - want + 180) % 360 - 180)
                end = a if diff_a < 90 else b
                thresholds[r] = (end['lat'], end['lon'])

        elif aero == 'parking_position':
            ref = tags.get('ref') or tags.get('name')
            if not ref:
                continue
            if el['type'] == 'node':
                lat, lon = el['lat'], el['lon']
            elif geom:
                lat = np.mean([p['lat'] for p in geom])
                lon = np.mean([p['lon'] for p in geom])
            else:
                continue
            # one record can carry several designators: "A12;A13"
            for part in re.split(r'[;,]', str(ref)):
                s = norm_stand(part)
                if s:
                    stands.setdefault(s, (lat, lon))

    return g, stands, thresholds


def distances_for_airport(apt, data):
    g, stands, thr = parse_airport(apt, data)
    print(f'  {apt}: {len(g.adj)} nodes, {len(stands)} stands, {len(thr)} thresholds')
    if not stands or not thr:
        return []

    rows = []
    for r, (rlat, rlon) in thr.items():
        node = g.nearest(rlat, rlon, max_m=600)
        dist = g.dijkstra(node) if node else {}
        for s, (slat, slon) in stands.items():
            sn = g.nearest(slat, slon, max_m=400)
            prava = meters(slat, slon, rlat, rlon)
            mreza = dist.get(sn, np.nan) if sn else np.nan
            # an absurdly long path means the network is broken there
            if np.isfinite(mreza) and mreza > 6 * prava + 2000:
                mreza = np.nan
            rows.append((apt, s, r, mreza, prava))
    return rows


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(OUT):
        print(f'{OUT} already exists; delete it to recompute')
        return

    all_rows = []
    for apt in AIRPORTS:
        print(f'{apt} ...')
        data = fetch(apt)
        if data is None:
            continue
        all_rows += distances_for_airport(apt, data)
        time.sleep(3)

    df = pd.DataFrame(all_rows, columns=['apt', 'stand_n', 'runway_n',
                                         'dist_taxi_m', 'dist_prava_m'])
    df['dist_taxi_m'] = df.dist_taxi_m.astype('float32')
    df['dist_prava_m'] = df.dist_prava_m.astype('float32')
    df.to_parquet(OUT, index=False)
    print(f'\nsaved {OUT}: {len(df)} pairs')
    print(df.groupby('apt').agg(pairs=('stand_n', 'size'),
                                with_path=('dist_taxi_m', lambda s: int(s.notna().sum())),
                                median_m=('dist_taxi_m', 'median')).round(0).to_string())


if __name__ == '__main__':
    main()
