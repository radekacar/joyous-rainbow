# Taxi-out time at ten European airports — PRC Data Challenge 2026

Team **joyous-rainbow**. Single-file LightGBM pipeline, plus the measurements
that decided what went into it and what was thrown out.

## What the task actually is

The challenge asks for `TAXITIME_SEC_mvt` of departures already flown, at
EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.

This is **post-operations estimation, not forecasting**. Every prediction is
made with the take-off time of the flight itself in hand, and with the airport's
whole movement log around it. A model that predicts taxi-out *before pushback* —
the problem an air traffic controller or an airline dispatcher has — gets none of
that, and the numbers here say nothing about how well such a model would do. The
distinction matters, because the single most informative feature in this solution
is one that forecasting cannot use.

In the data, `TAXITIME_SEC_mvt` equals `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`
for **100.00 %** of training departures: take-off minus off-block, exactly, with
no exceptions in 2,085,047 rows. The off-block field is blanked for departures in
the prediction set. So the target is one subtraction away from a field that was
removed, and the task is to reconstruct it.

The Network Manager data gives a second, independent measurement of the same
event: `AOBT_3_flt`, EUROCONTROL's own actual off-block time. Predicting
`MVT_TIME - AOBT_3` directly — using one system's measurement in place of the
other's — gives RMSE **428.2 s**, and **413.6 s** after a per-airport offset. The
two systems agree within 30 seconds on only 10.3 % of flights, and within ten
minutes on 92.4 %.

**The real problem, then, is not predicting how long an aircraft taxis. It is
predicting how much the airport's clock disagrees with EUROCONTROL's about the
same pushback.** The model takes that 428 s disagreement down to 229 s.

## Results

Validation holds out **January and July 2025** and trains on the remaining ten
months. A random K-fold would put flights from the same day, stand and runway on
both sides of the split and return an optimistic number that does not survive
contact with 2026 data.

| | without stand occupancy | with stand occupancy |
|---|---|---|
| overall RMSE (full pipeline) | 324.1 | **321.4** |
| main model, rows with an NM record | 232.5 | **228.7** |
| group A — no `AOBT_3` (n = 5,373) | 1822.5 | 1822.5 |
| group B — long, valid `AOBT_3` (n = 4,550) | 1114.5 | 1090.4 |
| group C — regular (n = 334,496) | 194.7 | **191.9** |

Official score in the open phase, for the configuration without stand
occupancy and with a single seed: **299.2342**.

### The error is not spread evenly

| group | rows | share of rows | share of squared error |
|---|---|---|---|
| A — no Network Manager record at all | 5,373 | 1.6 % | **50.2 %** |
| B — valid `AOBT_3`, recorded taxi-out over 40 min | 4,550 | 1.3 % | 15.2 % |
| C — everything else | 334,496 | 97.1 % | 34.6 % |

By month and group, which is the table that says where the metric really lives:

```
                      A                  B                  C          all
January    2907 (34.1%)       797 ( 3.7%)       181 (13.9%)          346
July       1206 (16.1%)      1288 (11.5%)       200 (20.8%)          300
```

**Group A in January alone is 34 % of the whole metric**, and inside it, two
rows at Paris carry most of that. See [docs/DATA.md](docs/DATA.md).

## What is in the model

Sixty-eight features. The ones that matter, in order of what they were measured
to be worth:

**Linear leaves** (`linear_tree=True`, `linear_lambda=1.0`) — the largest single
gain in the whole solution, local RMSE 364.0 → 338.5 and official 333.1 → 301.8.
A constant-leaf tree cannot predict a value above the largest one in its training
data, and in this task the error lives precisely above that ceiling. Trees with
linear models in the leaves extrapolate; trees with constants cannot.

**Stand occupancy** (`occ_h`, `occ_n_24h`) — hours since an *arriving* aircraft
last went on-block at the same stand, and how many arrivals occupied it in the
previous 24 hours. Both inputs survive into the prediction set: the departure's
take-off time, and arrivals' on-block times, which are present for all 670,407
arrivals there. Worth 4.35 s on rows with an NM record, measured at matched
round counts. Median taxi-out by occupancy bucket runs 1137 s under half an
hour, 693 s between half an hour and an hour, 849 s to ninety minutes, 901 s
between six and twelve — a 444 s spread that no other feature carried.

**A three-model treatment of group A** — a per-airport cubic-plus-hinge curve,
a linear-leaf LightGBM, and a classifier mixture, combined
`0.25·curve + 0.25·trees + 0.5·mixture`. Within the group: curve 1908, trees
1947, mixture 1861, the combination **1822**. Every alternative weighting
measured worse; see [docs/FINDINGS.md](docs/FINDINGS.md).

**Visibility** — of all weather variables, only visibility measurably helps
(4.5 s). Low-visibility procedures slow every ground movement in the period.
Temperature and precipitation act at the level of a whole day and add nothing
the date features do not already carry.

**Target encoding of stand and runway**, computed on the training split only.

**Stand-to-threshold taxiway distance** from OpenStreetMap: the apron and
taxiway graph pulled from the Overpass API, then Dijkstra from each stand to
each runway threshold. 96 % of stand names matched; `check_osm.py` reports the
rate per airport.

**Five-seed averaging** — `bagging_fraction` and `feature_fraction` make every
fit random, and averaging n fits removes a (1 − 1/n) share of that variance.
Measured: one seed 246.82, three seeds 245.27, so five reach 244.96. About
0.2 s on the official metric. Small, free, and it cannot hurt in expectation.

## Negative results

These cost most of the project's time, and they are the part most likely to be
useful to someone else. Full numbers in [docs/FINDINGS.md](docs/FINDINGS.md).

1. **Counting movements in a time window does not measure congestion.** Seven
   independent attempts — departures, all movements, per runway, several window
   lengths — landed between −0.8 and +1.8 s. The reason is that a count inside a
   window measures *demand*, while queueing delay follows the rate of *service*.
   The gap between consecutive take-offs on a runway does measure service, and
   separates median taxi-out from 1320 s at a 45–60 s mean gap down to 837 s
   above 400 s. It still did not help the model, for the reason in point 2.
2. **Four ground-sequence features made the model worse**, by 4.56 s at a
   matched round count, while costing 6.8× the training time. `gap_rwy` is
   partly determined by the take-off time of the flight being predicted, so it
   measures an outcome rather than a cause — the same trap as `proxy`.
3. **The apparent −738 s bias in group B is a regression fallacy.** Group B is
   defined by `y > 2400`, which conditions on the truth; under that conditioning
   even a perfect conditional-mean estimator looks shrunk toward the centre.
   Conditioned on the *prediction*, which is all that exists at prediction time,
   the model is already calibrated: `E[y | ŷ] − ŷ` is about +10 s across the
   bulk and turns **negative** in the top bin. Isotonic recalibration on
   1,723,531 out-of-fold rows made it much worse (242.2 → 267.4), linear
   recalibration slightly worse, and every multiplicative correction that helped
   group B cost more in group C.
4. **Isotonic calibration of the group A classifier gained nothing.** The Brier
   score improved, 0.0404 → 0.0390, and RMSE did not: 1825 against 1822.
5. **Every procedure that softens extremes hurt.** RMSE's optimum is the
   conditional mean, so a robust (Huber) fit on group A measured 2178 against
   1908, and clipping the training target at 3600 s instead of 10800 s cost
   10 s.
6. **Hedging rare extremes is mathematically futile.** For a fraction q of rows
   that jump by Δ, the best constant shift reduces MSE by a factor of only
   (1 − q). With q ≈ 0.002 there is nothing to win; such rows must be
   *identified*, not insured against.
7. **The 2024-challenge tail hypothesis was dropped.** An earlier draft claimed
   the 2026 tail was milder than 2025's. That was never measured and has been
   removed rather than restated.

## Two methodological mistakes, kept on the record

**A contaminated A/B.** Five feature sets were compared while early stopping
chose the number of boosting rounds separately in each, from 603 to 1586. The
arm with the most rounds had the best RMSE and the arm with the fewest had
nearly the worst, so the comparison measured two things at once and attributed
both to the features. `sweep.py` exists to fix this: it trains each arm once to
a fixed 2600 rounds with early stopping off, then scores every arm at the same
grid of round counts using `num_iteration` on the same booster. The corrected
table is in [docs/FINDINGS.md](docs/FINDINGS.md). It also showed the round count
barely matters above 1500 — 0.64 s across 1100 rounds — while below it the curve
falls away by 4 s.

**Microseconds, twice.** The timestamps are microseconds. Dividing by `10**9`
instead of `10**6` compresses every interval by a factor of a thousand, which
turns a one-hour congestion window into seven days and a median stand occupancy
of 1.68 h into six seconds. It happened once in the congestion features, where
it invalidated a published conclusion that congestion was useless, and again in
the occupancy feature, where it made a working feature look empty. Both are
flagged in the code where the divisor appears.

## Reproducing this

```bash
pip install pandas numpy pyarrow lightgbm requests
mkdir data                      # put the challenge parquet files here
python osm_distances.py         # optional, writes cache/stand_dist.parquet
python taxiout.py               # weather, features, validation, submission
```

`taxiout.py` runs in four steps and caches aggressively: METAR weather once,
features once per code version, then validation and the submission. On sixteen
cores the whole run is about 25 minutes, most of it the five seeds.

The settings at the top of the file are the knobs. `DATA_DIR`, `TEAM_NAME`,
`RANK_FILE` and `SUB_FILE` point at the data; `SEEDS`, `BEST_ITER`,
`USE_OCCUPANCY` and `VALID_MONTHS` change the model. `RUN_OPTUNA = True`
re-runs the 100-trial hyperparameter search, which takes hours; its result is
hard-coded as the default so the published numbers reproduce without it.

```bash
python sweep.py                 # feature sets × round counts, matched
python finalize.py              # name the final file and verify it
python upload.py <file>         # upload, then read the bucket back
```

| file | what it does |
|---|---|
| `taxiout.py` | the whole pipeline: weather, features, both models, submission |
| `sweep.py` | compares feature sets at matched round counts |
| `finalize.py` | renames the chosen submission and runs nine validity checks |
| `upload.py` | uploads and verifies, reading credentials from outside the file |
| `osm_distances.py` | OpenStreetMap taxiway graph → stand-to-threshold distances |
| `check_osm.py` | reports the stand-name match rate per airport |
| `figures.py` | the figures, from the feature cache |

### Documentation

| file | what is in it |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | the pipeline, the split, the features, the hyperparameters, the caching |
| [docs/FINDINGS.md](docs/FINDINGS.md) | every measurement, positive and negative, with the tables |
| [docs/DATA.md](docs/DATA.md) | the two data-quality mechanisms and the rows that cannot be predicted |

### The data is not here

The challenge dataset is not redistributable under its licence, so `data/` is in
`.gitignore` and nothing derived from it is committed. The weather comes from the
Iowa Environmental Mesonet METAR archive and the airport layouts from
OpenStreetMap; both are fetched by the scripts.

### Two traps worth knowing about before you run it

`MVT_ID_mvt` must not be stored as `float32`. The identifiers have nine digits
and `float32` holds about seven, so distinct flights collapse onto one id — it
once turned 344,841 unique identifiers into 22,048 and produced a submission
that was silently invalid. `finalize.py` checks the dtype directly, because a
round-trip test does not catch it: once collapsed, the values agree with
themselves.

Credentials belong in `keys.txt` or in environment variables, never in a
committed file. Keys that have been pasted anywhere should be rotated at
<https://s3-console.opensky-network.org/access-keys>.

## Attribution of ideas

Two ideas came from the public write-up of team **kind-mango** rather than from
this work: treating the rows with no Network Manager record as a separate
population, and using the schedule-to-take-off difference as the main predictor
inside that population. The implementation here — the three-model combination,
the per-airport curve and the classifier mixture — is this project's own, but the
framing is not, and it would be dishonest to present it as such.

The decision to use `AOBT_3_flt` at all deserves a note. It is EUROCONTROL's
actual off-block time, which means the feature `proxy = MVT_TIME - AOBT_3` is a
measurement of the same physical event as the target. It is in the provided data
and the rules permit it, but anyone reading these numbers as evidence about
operational taxi-out prediction should know that this one feature carries most of
the signal, and that a forecasting system would not have it.

## Licence

GNU General Public License v3.0 only. See [LICENSE](LICENSE).
