"""
PRC Data Challenge 2026 - one clean measurement, no submission file
===================================================================
Why this exists.

Every A/B so far compared feature sets whose models had been stopped at
DIFFERENT numbers of boosting rounds, because early stopping chose the round
count separately in each arm:

    all 71 features (occ + sequence)   RMSE 234.80   at  603 rounds
    base 66 features                   RMSE 232.50   at 1220 rounds
    base + occ_h                       RMSE 231.04   at 1036 rounds
    base + 4 sequence features         RMSE 237.06   at 1193 rounds
    base + occ_h + occ_n_24h           RMSE 228.70   at 1586 rounds

Across those five, the arm with the most rounds also had the best RMSE and the
arm with the fewest had nearly the worst. Two effects were being measured at
once, so none of the differences could be attributed to the features.

This script separates them. Each arm is trained ONCE to MAX_ROUNDS with early
stopping switched off, and then scored at every round count in GRID using
num_iteration on the same booster. One fit per arm, a full curve per arm, and
every arm read at identical round counts.

It answers both open questions in a single run:
  - which feature set is better, at matched rounds
  - what BEST_ITER should be

It writes no submission file. Set the winner in the main script afterwards.

    python sweep.py

Licence: GNU GPL v3.0 only.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# taxiout.py supplies the data preparation, the features and the split, so
# this script cannot drift away from the pipeline it is measuring.
import taxiout as T

MAX_ROUNDS = 2600
GRID = [600, 900, 1200, 1500, 1800, 2100, 2400, 2600]


def main():
    print('=' * 72)
    print('loading features through taxiout_occ')
    print('=' * 72)
    T.collect_weather()
    train, _rank = T.prepare()

    tr, va = T.split_time(train)
    nmk = va.has_nm.values
    yk = va.TAXITIME_SEC_mvt.values
    print(f'training {len(tr)}, validation {len(va)}, of which {nmk.sum()} with NM')

    params = None
    pj = f'{T.CACHE_DIR}/best_params.json'
    if os.path.exists(pj):
        params = json.load(open(pj))['params']
        print('using the hyperparameters from the Optuna search')

    base = [f for f in T.MAIN_FEATS if f not in T.OCC_FEATS]
    arms = [
        ('base, no new features', base),
        ('+ occ_h', base + ['occ_h']),
        ('+ occ_h + occ_n_24h', base + ['occ_h', 'occ_n_24h']),
    ]
    arms = [(n, f) for n, f in arms if all(c in va.columns for c in f)]

    results = {}
    for name, feats in arms:
        print(f'\n--- {name}  ({len(feats)} features) ---', flush=True)
        import time
        t0 = time.time()
        # va=None means no early stopping: the model runs the full MAX_ROUNDS,
        # which is what makes the arms comparable.
        m = T.train_main(tr, None, params, rounds=MAX_ROUNDS, feats=feats)
        print(f'    trained {MAX_ROUNDS} rounds in {time.time() - t0:.0f}s', flush=True)
        row = {}
        for n in GRID:
            p = np.clip(m.predict(va[feats], num_iteration=n), 0, None)
            row[n] = T.rmse(p[nmk], yk[nmk])
            print(f'    {n:5d} rounds: RMSE on NM rows {row[n]:8.3f}', flush=True)
        results[name] = row

    print('\n' + '=' * 72)
    print('RMSE on rows with a Network Manager record, at matched round counts')
    print('=' * 72)
    hdr = f'{"":24s}' + ''.join(f'{n:>9d}' for n in GRID)
    print(hdr)
    for name, row in results.items():
        print(f'{name:24s}' + ''.join(f'{row[n]:9.2f}' for n in GRID))

    print('\nbest cell per arm:')
    for name, row in results.items():
        n = min(row, key=row.get)
        print(f'  {name:24s} {row[n]:8.2f} at {n} rounds')

    allcells = [(r[n], n, nm) for nm, r in results.items() for n in GRID]
    v, n, nm = min(allcells)
    print(f'\noverall best: {nm} at {n} rounds, RMSE {v:.2f}')
    print(f'set BEST_ITER = {int(round(n / 1.1))} so that run_submit trains {n} rounds,')
    print('and switch the feature set to the winning arm.')
    json.dump({k: {str(a): b for a, b in v.items()} for k, v in results.items()},
              open(f'{T.CACHE_DIR}/sweep.json', 'w'), indent=1)
    print(f'\nsaved {T.CACHE_DIR}/sweep.json')


if __name__ == '__main__':
    main()
