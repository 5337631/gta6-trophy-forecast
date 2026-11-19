# GTA VI trophy rarity forecast

A forecasting program for make.fun's `gta6-trophy-rarity-forecast-01`
bounty. For each case it reads the recorded history of GTA VI's Exophase
trophy page (every trophy's earn rate and the tracked-player count) and
writes, for each of the 55 trophies, five quantiles (5%, 25%, 50%, 75%,
95%) of its earn rate at the case's target time. Scoring is pinball loss
on the logit scale, so a good entry needs an accurate median *and* an
honest spread.

## How it works

A rate is `earners / players`, and the numerator and denominator move for
different reasons, so the program forecasts them separately instead of
extrapolating the rate directly.

**History handling.** Exophase refreshes a game's counts only every few
hours; between refreshes every read is identical. The history is first
collapsed to the *distinct* reads — the ones where the player count or a
rate actually changed — and all fitting uses only those.

**Players.** The tracked-player count is extrapolated two ways and the
results are blended in log space: a damped linear trend fitted on the
recent distinct reads (damping 0.80 per day, because launch growth decays
quickly), and a log-time saturation curve, `log(players)` against
`ln(1 + hours since release)`, which flattens on its own. The forecast is
never below the last observed count.

**Earners.** For each trophy, `log(earners + 0.5)` is fitted on
`log(players)` and `ln(1 + hours since release)` over the distinct reads
(a single-predictor elasticity fit when there are too few reads).
Coefficients are clipped to sane ranges and shrunk 30% toward the median
coefficients of the trophy's tier (platinum / gold / silver / bronze),
since single-trophy fits over a few stepped reads are noisy; the platinum
trophy's player elasticity is additionally capped, because completing
every trophy barely scales with new arrivals in week one. The intercept
is re-anchored after shrinking so the model still passes near the last
observed earner count. Earner forecasts are floored at the last observed
count — players do not un-earn trophies.

**Median and spread.** The median rate is forecast earners over forecast
players, computed and shifted in logit space with a cap on how far it may
move from the last rate. The spread (in logit units) combines horizon
growth, the trophy's own step volatility across distinct reads, the fit
residual, and a staleness term for when the current page values have sat
unchanged much longer than the typical refresh gap; it is clamped to
[0.30, 1.35]. Two special cases matter:

- A *flat history* (at most two distinct reads, as on launch night) means
  the next refresh jump is completely unseen. The spread then has a floor
  of 0.90 and the median is nudged down slightly, because a refresh mostly
  dilutes rates: the player count jumps far more than most earner counts.
- A *saturated trophy* (last rate ≥ 99.9%) is held at a 99.5% median with
  a wide spread, instead of pinning all five quantiles at the 99.99 clamp.

**Safety.** Every trophy is forecast inside its own guard: any failure
falls back to a persistence forecast (last rate, random-walk spread), so
the program always emits a valid file. Output is deterministic — the
same case always produces byte-identical output.

## Running

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. In the grader the
directories default to `/input` and `/output`.

## Local validation

- Both practice cases from the competition kit (`p00`, `p01`) run in
  ~0.07 s each (limit: 8 s), produce 55 valid rows each, and are
  byte-identical across repeated runs. Their histories are completely
  flat, so the flat-history path is what fires: medians sit 0.30 logit
  below the last rates and the 90% band is ~2.96 logit wide, versus the
  persistence baseline's 0.41 — the practice answers sit on the far side
  of a refresh jump, which is what that width is for.
- A synthetic final-like case (stepped refreshes over 2.5 days, player
  count growing 638 → ~18,700, horizon 4.52 days) also validates: the
  medians move down by ~0.19 logit on average as dilution continues,
  with the platinum trophy moving most, and runtime stays under 0.1 s.
