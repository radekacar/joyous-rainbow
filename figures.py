"""
Figures for the write-up
========================

Builds five figures from the cache left by taxiout_final.py:

  fig1_distribution.png   taxi-out distribution and its tail
  fig2_groups.png         share of flights versus share of error, by group
  fig3_group_a.png        delay to taxi-out relation in group A, with the curve
  fig4_congestion.png     congestion correlations per airport and by quintile
  fig5_progress.png       progression of the result

Usage:
    python figures.py                 # uses ./cache
    python figures.py /path/to/cache

Requires: pandas numpy matplotlib pyarrow
Licence: GNU GPL v3.0 only.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CACHE = sys.argv[1] if len(sys.argv) > 1 else 'cache'
OUT = 'figures'
VALID_MONTHS = [1, 7]
DPI = 200

# one consistent look for every figure
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': DPI, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'figure.autolayout': True,
})
C1, C2, C3, C4 = '#2E5E8C', '#C25E4A', '#6A8F5F', '#8A7BA8'


def load():
    p = f'{CACHE}/train.parquet'
    if not os.path.exists(p):
        raise SystemExit(f'{p} missing. Run taxiout_final.py first.')
    return pd.read_parquet(p)


def groups(d):
    """A - no AOBT_3, B - long with AOBT_3 (>40 min), C - regular."""
    a = ~d.has_nm.values
    b = (~a) & (d.TAXITIME_SEC_mvt.values > 2400)
    return a, b, (~a) & (~b)


# ---------------------------------------------------------------- figure 1
def fig_distribution(d):
    y = d.TAXITIME_SEC_mvt.dropna().values / 60.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.hist(y[y <= 60], bins=120, color=C1, edgecolor='none')
    ax1.axvline(np.median(y), color=C2, lw=1.5, ls='--',
                label=f'median {np.median(y):.1f} min')
    ax1.set_xlabel('taxi-out (minutes)')
    ax1.set_ylabel('flights')
    ax1.set_title('Distribution up to 60 minutes (98.9 % of flights)')
    ax1.legend(frameon=False)

    tail = y[y > 40]
    ax2.hist(tail, bins=np.logspace(np.log10(40), np.log10(max(tail.max(), 60)), 60),
             color=C2, edgecolor='none')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('taxi-out (minutes, log scale)')
    ax2.set_ylabel('flights')
    ax2.set_title(f'Tail: {len(tail)} flights above 40 minutes')
    fig.savefig(f'{OUT}/slika1_raspodjela.png')
    plt.close(fig)
    print('figure 1 done')


# ---------------------------------------------------------------- figure 2
def fig_groups(d, pred=None):
    """Udio letova naspram udjela greske. Ako nema predikcija, koristi se
    doprinos same varijanse (kvadrirano odstupanje od medijana)."""
    va = d[d.month.isin(VALID_MONTHS)]
    y = va.TAXITIME_SEC_mvt.values
    a, b, c = groups(va)
    e2 = (pred - y) ** 2 if pred is not None else (y - np.median(y)) ** 2

    share_n = [m.mean() * 100 for m in (c, a, b)]
    share_e = [e2[m].sum() / e2.sum() * 100 for m in (c, a, b)]
    names = ['C  regular', 'A  no NM record', 'B  long with NM record']

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ypos = np.arange(3)
    ax.barh(ypos + 0.2, share_n, height=0.38, color=C1, label='share of flights')
    ax.barh(ypos - 0.2, share_e, height=0.38, color=C2, label='share of total error')
    for i, (n, e) in enumerate(zip(share_n, share_e)):
        ax.text(n + 1, i + 0.2, f'{n:.1f} %', va='center', fontsize=9)
        ax.text(e + 1, i - 0.2, f'{e:.1f} %', va='center', fontsize=9)
    ax.set_yticks(ypos, names)
    ax.set_xlabel('percent')
    ax.set_xlim(0, 108)
    ax.set_title('A minority of flights carries most of the error')
    ax.legend(frameon=False, loc='lower right')
    fig.savefig(f'{OUT}/slika2_grupe.png')
    plt.close(fig)
    print('figure 2 done')


# ---------------------------------------------------------------- figure 3
def fig_group_a(d, apt='LIRF'):
    """Veza kasnjenja i taxi-out vremena u grupi A, sa krivom iz modela."""
    tr = d[~d.month.isin(VALID_MONTHS)]
    sub = tr[(~tr.has_nm) & tr.d_sched.notna() & (tr.apt == apt)]
    if len(sub) < 50:
        print(f'figure 3 skipped: too few rows for {apt}')
        return
    x, yy = sub.d_sched.values / 3600, sub.TAXITIME_SEC_mvt.values / 3600
    K = 3.0                                   # prelom na 3 sata

    b3 = np.polyfit(x, yy, 3)
    X = np.c_[x, np.maximum(x - K, 0), np.ones(len(x))]
    bp = np.linalg.lstsq(X, yy, rcond=None)[0]
    bl = np.polyfit(x, yy, 1)
    g = np.linspace(x.min(), np.percentile(x, 99.5), 200)
    kriva = 0.5 * np.polyval(b3, g) + 0.5 * (bp[0] * g + bp[1] * np.maximum(g - K, 0) + bp[2])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    tacno = np.abs(sub.d_sched.values - sub.TAXITIME_SEC_mvt.values) < 60
    ax.scatter(x[~tacno], yy[~tacno], s=7, alpha=.35, color=C1, label='other flights')
    ax.scatter(x[tacno], yy[tacno], s=7, alpha=.55, color=C3,
               label='taxi = delay (scheduled time written in)')
    ax.plot(g, np.polyval(bl, g), color='0.45', lw=1.6, ls='--', label='straight line')
    ax.plot(g, kriva, color=C2, lw=2.2, label='model curve (cubic + hinge)')
    ax.set_xlabel('take-off delay against the schedule (hours)')
    ax.set_ylabel('recorded taxi-out (hours)')
    ax.set_title(f'Group A at {apt}: taxi-out measures delay, not taxiing')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.savefig(f'{OUT}/slika3_grupa_a.png')
    plt.close(fig)
    print('figure 3 done')


# ---------------------------------------------------------------- figure 4
def fig_congestion(d):
    ok = d[d.has_nm & d.TAXITIME_SEC_mvt.between(60, 10800)]
    glob = ok.n_dep_1800.corr(ok.TAXITIME_SEC_mvt, method='spearman')
    per = ok.groupby('apt').apply(
        lambda g: g.n_dep_1800.corr(g.TAXITIME_SEC_mvt, method='spearman')
    ).sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.barh(per.index, per.values, color=C1)
    ax1.axvline(glob, color=C2, lw=1.8, ls='--',
                label=f'all airports pooled: {glob:+.2f}')
    ax1.axvline(per.median(), color=C3, lw=1.8, ls=':',
                label=f'median within airport: {per.median():+.2f}')
    ax1.axvline(0, color='0.3', lw=0.8)
    ax1.set_xlabel('Spearman correlation of departure count and taxi-out')
    ax1.set_title('Pooling airports weakens the relation')
    ax1.legend(frameon=False, fontsize=8.5, loc='lower right')

    q = pd.qcut(ok.n_rwy_1800, 5, labels=['1 lowest', '2', '3', '4', '5 highest'],
                duplicates='drop')
    med = ok.groupby(q).TAXITIME_SEC_mvt.median() / 60
    ax2.bar(med.index.astype(str), med.values, color=C1)
    ax2.bar([med.index.astype(str)[-1]], [med.values[-1]], color=C2)
    for i, v in enumerate(med.values):
        ax2.text(i, v + .15, f'{v:.1f}', ha='center', fontsize=9)
    ax2.set_ylabel('median taxi-out (minutes)')
    ax2.set_xlabel('runway congestion quintile (+/-30 min)')
    ax2.set_title('Congestion bites only in the densest quintile')
    fig.savefig(f'{OUT}/slika4_simpson.png')
    plt.close(fig)
    print('figure 4 done')


# ---------------------------------------------------------------- figure 5
def fig_progress():
    koraci = ['constant\n(median)', 'LightGBM\nalone', '+ group A\n(straight line)',
              '+ longer\ntraining range', '+ congestion\nand regime',
              '+ cubic\ncurve', 'official\n2026 score']
    vr = [520, 531, 382, 372, 371, 364, 337]
    boje = [ '0.6', '0.6', C1, C1, C1, C1, C3]

    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.bar(range(len(vr)), vr, color=boje)
    for i, v in enumerate(vr):
        ax.text(i, v + 6, str(v), ha='center', fontsize=9.5)
    ax.set_xticks(range(len(vr)), koraci, fontsize=8.5)
    ax.set_ylabel('RMSE (seconds)')
    ax.set_ylim(0, 600)
    ax.set_title('Progress; the first bars are local validation on 2025')
    fig.savefig(f'{OUT}/slika5_napredak.png')
    plt.close(fig)
    print('figure 5 done')


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    print(f'{len(d)} departures loaded')

    pred = None
    npy = f'{CACHE}/pred_valid.npy'
    if os.path.exists(npy):
        pred = np.load(npy)
        print('validation predictions loaded')

    fig_distribution(d)
    fig_groups(d, pred)
    fig_group_a(d)
    fig_congestion(d)
    fig_progress()
    print(f'\nall figures are in {OUT}/')


if __name__ == '__main__':
    main()
