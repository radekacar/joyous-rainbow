"""
PRC Data Challenge 2026 - make the final submission file and verify it
======================================================================
The final phase accepts ONE file, and the organisers prescribed the name:

    <team-name>_final.parquet      ->      joyous-rainbow_final.parquet

Just run it. No arguments.

    python finalize.py

It copies SOURCE to that name and then re-runs every validity check against
final_submitting.parquet, so the file that gets uploaded is verified under the
name it will be uploaded as, not under the name it was written as.

Checks: row count, identifier order, identifier uniqueness, identifier dtype,
column names and order, missing values, infinite values, negative values.

The dtype check earns its place. float32 holds about seven significant digits
and these identifiers have nine, so storing them as float32 silently collapses
distinct flights onto one id - it once turned 344,841 unique ids into 22,048. A
round-trip test does NOT catch it, because once collapsed the values agree with
themselves. The dtype itself has to be checked.

Licence: GNU GPL v3.0 only.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings --
TEAM_NAME = 'joyous-rainbow'
SOURCE    = 'joyous-rainbow_v1.parquet'    # whatever taxiout.py last wrote
DATA_DIR  = r'data'
SUB_FILE  = 'final_submitting.parquet'
# ---------------------------------------------------------------------------

TARGET = f'{TEAM_NAME}_final.parquet'


def main():
    # the script lives next to the data, so run from its own folder whatever
    # the working directory happened to be when it was launched
    here = os.path.dirname(os.path.abspath(__file__))
    if here:
        os.chdir(here)
    print(f'working in {os.getcwd()}\n')

    src = SOURCE
    if '--from' in sys.argv:
        src = sys.argv[sys.argv.index('--from') + 1]

    if not os.path.exists(src):
        have = sorted(f for f in os.listdir('.') if f.endswith('.parquet'))
        raise SystemExit(f'source not found: {src}\n'
                         f'parquet files here: {have}\n'
                         f'set SOURCE at the top of this file, or pass '
                         f'--from <file>')
    tpl_path = os.path.join(DATA_DIR, SUB_FILE)
    if not os.path.exists(tpl_path):
        raise SystemExit(f'template not found: {os.path.abspath(tpl_path)}\n'
                         f'set DATA_DIR at the top of this file')

    shutil.copyfile(src, TARGET)
    print(f'copied  {src}  ->  {TARGET}')

    sub = pd.read_parquet(TARGET)
    tpl = pd.read_parquet(tpl_path)

    checks = []

    def ok(name, cond, detail=''):
        checks.append((name, bool(cond), detail))

    ok('file name', TARGET == f'{TEAM_NAME}_final.parquet', TARGET)
    ok('row count', len(sub) == len(tpl), f'{len(sub)} vs {len(tpl)} expected')
    ok('columns', list(sub.columns) == ['MVT_ID_mvt', 'TAXITIME_SEC_mvt'],
       str(list(sub.columns)))
    same = (len(sub) == len(tpl) and
            (sub.MVT_ID_mvt.astype('int64').values
             == tpl.MVT_ID_mvt.astype('int64').values).all())
    ok('identifiers match, in order', same)
    ok('identifiers unique', sub.MVT_ID_mvt.nunique() == len(sub),
       f'{sub.MVT_ID_mvt.nunique()} unique of {len(sub)}')
    ok('identifier dtype is not float32', sub.MVT_ID_mvt.dtype != np.float32,
       str(sub.MVT_ID_mvt.dtype))
    ok('no missing predictions', sub.TAXITIME_SEC_mvt.notna().all(),
       f'{int(sub.TAXITIME_SEC_mvt.isna().sum())} missing')
    ok('all finite', np.isfinite(sub.TAXITIME_SEC_mvt).all())
    ok('all positive', (sub.TAXITIME_SEC_mvt > 0).all(),
       f'min {sub.TAXITIME_SEC_mvt.min():.1f}')

    print()
    width = max(len(n) for n, _, _ in checks)
    for name, good, detail in checks:
        print(f'  [{"ok  " if good else "FAIL"}] {name:<{width}}  {detail}')

    bad = [n for n, g, _ in checks if not g]
    print()
    if bad:
        print('=' * 62)
        print('DO NOT UPLOAD. Failed: ' + ', '.join(bad))
        print('=' * 62)
        raise SystemExit(1)

    print('=' * 62)
    print('every check passed. this is the file to upload:')
    print(f'  {os.path.abspath(TARGET)}')
    print(f'  {os.path.getsize(TARGET) / 1e6:.1f} MB, {len(sub)} rows')
    print('=' * 62)
    print('\ndistribution of the predictions, seconds:')
    print(sub.TAXITIME_SEC_mvt.describe().round(1).to_string())
    print(f'\nIn the MinIO console, open the bucket prc-2026-{TEAM_NAME} and use')
    print('"Upload File", NOT "Upload Folder" - a folder upload once put the file')
    print('in a subfolder with %2F in its name, and here the name must match exactly.')
    print(f'Afterwards the listing must show {TARGET} at the top level, no prefix.')
    print(f'\nOr with the mc client:')
    print(f'  mc cp "{TARGET}" <your-alias>/prc-2026-{TEAM_NAME}/"{TARGET}"')
    print(f'  mc ls <your-alias>/prc-2026-{TEAM_NAME}/')


if __name__ == '__main__':
    main()
