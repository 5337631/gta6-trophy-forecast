"""Forecast GTA VI trophy earn rates for make.fun (gta6-trophy-rarity-forecast-01).

The program reads one case at a time from $MAKEFUN_INPUT_DIR (default /input)
and writes one forecast per case to $MAKEFUN_OUTPUT_DIR (default /output).

Model, in one paragraph: a trophy's earn rate is earners / players, and the
two move for different reasons, so they are forecast separately.  The player
count is extrapolated with a damped trend blended with a log-time saturation
curve.  Each trophy's earner count is extrapolated with an elasticity fit of
log(earners) on log(players) and log(1 + hours since release), fitted only on
the distinct reads (Exophase refreshes its page in steps, so most reads are
exact repeats and carry no new information), with coefficients shrunk toward
the median of the trophy's tier.  The median forecast is the ratio of the two
forecasts; quantiles spread around it on the logit scale, with a per-trophy
spread driven by the horizon, the trophy's own step volatility, and how stale
the history is.  Anything unexpected falls back to persistence, so the
program always writes a valid forecast and never crashes.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE_HOUR_ZERO = "2026-11-19T00:00:00Z"


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds (float)."""
    if text is None:
        return None
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


# ---------------------------------------------------------------------------
# History preparation
# ---------------------------------------------------------------------------

def distinct_reads(history):
    """Collapse the stepped Exophase reads to the ones that actually changed.

    Between refreshes the page is byte-identical in every field we use, so a
    repeated read adds no information and would badly bias any fit toward
    "nothing ever moves".  Keep the first read, then every read whose player
    count or rate vector differs from the last kept read.
    """
    kept = []
    for read in history or []:
        if not isinstance(read, dict):
            continue
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or rates is None:
            continue
        if kept:
            prev = kept[-1]
            if prev.get("players_tracked") == players and prev.get("rates") == rates:
                continue
        kept.append(read)
    return kept


def read_time(read, fallback):
    t = parse_time(read.get("at"))
    return fallback if t is None else t


# ---------------------------------------------------------------------------
# Tiny weighted least squares (normal equations with a ridge)
# ---------------------------------------------------------------------------

def solve_linear(matrix, vector):
    """Solve a small dense system by Gaussian elimination; None if singular."""
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / piv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit_linear(xs, ys, weights, ridge=1e-6):
    """Fit y ~ intercept + sum(beta_j * x_j).  xs is a list of feature rows."""
    if not xs or len(xs) != len(ys):
        return None
    dim = len(xs[0]) + 1
    matrix = [[0.0] * dim for _ in range(dim)]
    vector = [0.0] * dim
    for row, y, w in zip(xs, ys, weights):
        feats = [1.0] + list(row)
        for i in range(dim):
            vector[i] += w * feats[i] * y
            for j in range(dim):
                matrix[i][j] += w * feats[i] * feats[j]
    for i in range(1, dim):
        matrix[i][i] += ridge
    beta = solve_linear(matrix, vector)
    if beta is None:
        return None
    residuals = []
    for row, y in zip(xs, ys):
        pred = beta[0] + sum(b * x for b, x in zip(beta[1:], row))
        residuals.append(y - pred)
    return beta, residuals


# ---------------------------------------------------------------------------
# Player-count forecast
# ---------------------------------------------------------------------------

def forecast_players(reads, checkpoint_ts, target_ts):
    """Forecast players_tracked at the target; returns (players, info dict).

    Two views are blended in log space:
      * a damped linear trend fitted on the recent distinct reads, and
      * a log-time saturation curve, log(players) ~ a + b * ln(1 + hours
        since release), which naturally flattens as the launch surge fades.
    With a single distinct read there is no trend to fit and the count is
    held flat (any refresh jump then belongs in the spread, not the median).
    """
    last_players = float(reads[-1].get("players_tracked") or 0)
    if last_players <= 0:
        return 1.0, {"method": "degenerate"}
    if len(reads) < 2:
        return last_players, {"method": "flat"}

    horizon_days = max(0.0, (target_ts - checkpoint_ts) / 86400.0)
    times = [read_time(r, checkpoint_ts) for r in reads]
    logp = [math.log(max(1.0, float(r.get("players_tracked") or 1))) for r in reads]

    # Recent window: the last 36 hours of distinct reads, at least 3 points.
    window_start = times[-1] - 36 * 3600.0
    idx = [i for i, t in enumerate(times) if t >= window_start]
    if len(idx) < 3:
        idx = list(range(max(0, len(reads) - 4), len(reads)))

    forecasts = []

    # Damped trend on log(players) against days.
    xs = [[(times[i] - times[idx[0]]) / 86400.0] for i in idx]
    ys = [logp[i] for i in idx]
    weights = [1.0 + 0.15 * k for k in range(len(idx))]
    fit = fit_linear(xs, ys, weights, ridge=1e-4)
    if fit is not None:
        beta, _ = fit
        slope = max(0.0, min(1.2, beta[1]))  # per day, in log space
        level = beta[0] + beta[1] * ((times[-1] - times[idx[0]]) / 86400.0)
        phi = 0.80  # damping per day: launch growth decays fast
        damped_days = sum(phi ** k for k in range(1, int(math.ceil(horizon_days)) + 1))
        if horizon_days < 1.0:
            damped_days = horizon_days * phi
        forecasts.append(level + slope * damped_days)

    # Log-time saturation curve over all distinct reads.
    release_ts = parse_time(RELEASE_HOUR_ZERO) or times[0]
    xs2 = [[math.log1p(max(0.0, (t - release_ts) / 3600.0))] for t in times]
    fit2 = fit_linear(xs2, logp, [1.0] * len(times), ridge=1e-4)
    if fit2 is not None:
        beta2, _ = fit2
        target_x = math.log1p(max(0.0, (target_ts - release_ts) / 3600.0))
        forecasts.append(beta2[0] + max(0.0, beta2[1]) * target_x)

    if not forecasts:
        return last_players, {"method": "flat-fallback"}

    log_pred = sum(forecasts) / len(forecasts)
    players = math.exp(log_pred)
    # Guardrails: never below the last count, never an absurd multiple of it.
    players = max(last_players, min(players, last_players * 60.0))
    return players, {"method": "blend", "count": len(forecasts)}


# ---------------------------------------------------------------------------
# Earner-count forecast per trophy
# ---------------------------------------------------------------------------

def earner_series(reads, index):
    """(times, players, earners) for one trophy across the distinct reads."""
    times, players, earners = [], [], []
    fallback_t = 0.0
    for read in reads:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        p = float(read.get("players_tracked") or 0)
        if rate is None or p <= 0:
            continue
        t = read_time(read, fallback_t)
        fallback_t = t
        times.append(t)
        players.append(p)
        earners.append(rate / 100.0 * p)
    return times, players, earners


def fit_earner_model(times, players, earners, release_ts):
    """Fit log(earners + 0.5) on log(players) and log1p(hours since release).

    Returns (beta, residual_std) or None when there is too little signal.
    Coefficients are clipped to sane ranges: earners cannot grow much faster
    than the player base for long, and neither predictor should run negative
    over a launch week.
    """
    n = len(times)
    if n < 3:
        return None
    ys = [math.log(e + 0.5) for e in earners]
    hours = [max(0.0, (t - release_ts) / 3600.0) for t in times]
    if n >= 5:
        xs = [[math.log(p), math.log1p(h)] for p, h in zip(players, hours)]
        weights = [1.0 + 0.1 * k for k in range(n)]
        fit = fit_linear(xs, ys, weights, ridge=1e-3)
        if fit is not None:
            beta, residuals = fit
            beta[1] = min(2.0, max(0.0, beta[1]))
            beta[2] = min(2.5, max(0.0, beta[2]))
            scale = max(1e-9, sum(r * r for r in residuals) / max(1, n - 3))
            return beta, math.sqrt(scale), 2
    # Single-predictor fallback: elasticity on the player count alone.
    xs = [[math.log(p)] for p in players]
    fit = fit_linear(xs, ys, [1.0] * n, ridge=1e-3)
    if fit is None:
        return None
    beta, residuals = fit
    beta[1] = min(2.0, max(0.0, beta[1]))
    scale = max(1e-9, sum(r * r for r in residuals) / max(1, n - 2))
    return [beta[0], beta[1], 0.0], math.sqrt(scale), 1


def predict_earners(beta, players_pred, target_ts, release_ts):
    hours = max(0.0, (target_ts - release_ts) / 3600.0)
    log_e = beta[0] + beta[1] * math.log(max(1.0, players_pred)) + beta[2] * math.log1p(hours)
    return max(0.0, math.exp(log_e) - 0.5)


# ---------------------------------------------------------------------------
# Per-case forecast
# ---------------------------------------------------------------------------

def persistence_row(trophy_id, last_rate, horizon_days):
    spread = 0.5 * math.sqrt(max(0.0, horizon_days))
    lg = logit(last_rate if last_rate is not None else 0.0)
    return {"id": trophy_id, "q": [round4(inv_logit(lg + z * spread)) for z in Z]}


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_days = horizon_minutes / 1440.0
    checkpoint_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if checkpoint_ts is None:
        checkpoint_ts = 0.0
    if target_ts is None:
        target_ts = checkpoint_ts + horizon_minutes * 60.0

    last_rates = []
    if history and isinstance(history[-1], dict):
        last_rates = history[-1].get("rates") or []

    def last_rate(i):
        if i < len(last_rates):
            return last_rates[i]
        return None

    reads = distinct_reads(history)
    if not reads or not trophies:
        return {"trophies": [persistence_row(t.get("id"), last_rate(i), horizon_days)
                             for i, t in enumerate(trophies)]}

    release_ts = parse_time(RELEASE_HOUR_ZERO) or read_time(reads[0], checkpoint_ts)
    players_pred, _info = forecast_players(reads, checkpoint_ts, target_ts)
    last_players = float(reads[-1].get("players_tracked") or 1)

    # First pass: fit every trophy's earner model so tier medians exist.
    fits = []
    for i in range(len(trophies)):
        times, players, earners = earner_series(reads, i)
        fit = fit_earner_model(times, players, earners, release_ts) if len(times) >= 3 else None
        fits.append((times, players, earners, fit))

    tier_betas = {}
    for i, trophy in enumerate(trophies):
        fit = fits[i][3]
        if fit is not None:
            tier_betas.setdefault(trophy.get("tier"), []).append(fit[0])
    tier_median_beta = {}
    for tier, betas in tier_betas.items():
        tier_median_beta[tier] = [median([b[k] for b in betas]) for k in range(3)]

    # Step volatility per trophy, on the logit scale, for the spread.
    def step_volatility(i):
        times, _players, _earners = fits[i][0], fits[i][1], fits[i][2]
        rates = []
        for read in reads:
            r = (read.get("rates") or [])
            rates.append(r[i] if i < len(r) else None)
        diffs = []
        for k in range(1, len(reads)):
            a, b = rates[k - 1], rates[k]
            if a is None or b is None:
                continue
            dt_days = max(1e-6, (read_time(reads[k], 0.0) - read_time(reads[k - 1], 0.0)) / 86400.0)
            diffs.append(abs(logit(b) - logit(a)) / math.sqrt(dt_days))
        if not diffs:
            return None
        return math.sqrt(sum(d * d for d in diffs) / len(diffs))

    volatilities = [step_volatility(i) for i in range(len(trophies))]
    known_vols = [v for v in volatilities if v is not None]
    median_vol = median(known_vols) if known_vols else 0.45

    # Staleness: how long the current page values have sat unchanged, and
    # how many genuinely different pages the history contains at all.  A
    # history with one distinct read (launch night's flat stretch) says
    # nothing about the size of the next refresh jump, so the spread must
    # carry that uncertainty instead of the fitted volatility, which is 0.
    distinct_times = [read_time(r, checkpoint_ts) for r in reads]
    flat_history = len(reads) <= 2
    stale_sigma = 0.0
    if len(reads) >= 3:
        gaps = [(distinct_times[k] - distinct_times[k - 1]) / 3600.0
                for k in range(1, len(distinct_times))]
        typical_gap = median(gaps) if gaps else 0.0
        stale_hours = max(0.0, (checkpoint_ts - distinct_times[-1]) / 3600.0)
        if typical_gap > 0 and stale_hours > 1.5 * typical_gap:
            stale_sigma = 0.20

    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id")
        lr = last_rate(i)
        try:
            row = forecast_one_trophy(
                i, trophy, fits[i], tier_median_beta, players_pred, last_players,
                target_ts, release_ts, lr, horizon_days,
                volatilities[i] if volatilities[i] is not None else median_vol,
                stale_sigma, flat_history,
            )
        except Exception:
            row = persistence_row(tid, lr, horizon_days)
        rows.append(row)
    return {"trophies": rows}


def forecast_one_trophy(index, trophy, fit_data, tier_median_beta, players_pred,
                        last_players, target_ts, release_ts, last_rate,
                        horizon_days, volatility, stale_sigma=0.0,
                        flat_history=False):
    times, players, earners, fit = fit_data
    tid = trophy.get("id")
    if last_rate is None:
        return persistence_row(tid, 0.0, horizon_days)

    # Saturated trophy (every tracked player already holds it): the logit
    # clamp would pin all five quantiles at 99.99 and score terribly if the
    # rate slips at all as new players arrive.  Hold the median just under
    # the ceiling and give it a genuinely wide spread instead.
    if last_rate >= 99.9:
        lg = logit(99.5)
        sigma = 1.0
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        return {"id": tid, "q": sorted(qs)}

    last_earners = last_rate / 100.0 * last_players if last_players > 0 else 0.0

    # Median from the earner model, shrunk toward the tier's median betas.
    q50 = None
    resid = 0.35
    if fit is not None:
        beta, resid, _nfeat = fit
        tier_beta = tier_median_beta.get(trophy.get("tier"))
        if tier_beta is not None:
            shrink = 0.30
            beta = [beta[0] * (1 - shrink) + tier_beta[0] * shrink,
                    beta[1] * (1 - shrink) + tier_beta[1] * shrink,
                    beta[2] * (1 - shrink) + tier_beta[2] * shrink]
        if trophy.get("tier") == "platinum":
            # A platinum means holding every other trophy; in week one its
            # earner count barely grows with the player base, whatever a
            # noisy short fit says.
            beta[1] = min(beta[1], 0.6)
            # Re-anchor the intercept so the shrunk model still passes near
            # the last observed earner count.
            if earners:
                pred_last = predict_earners(beta, last_players, times[-1], release_ts)
                if pred_last > 0:
                    beta[0] += math.log((earners[-1] + 0.5) / (pred_last + 0.5))
        earners_pred = predict_earners(beta, players_pred, target_ts, release_ts)
        earners_pred = max(earners_pred, last_earners)  # earners do not un-earn
        earners_pred = min(earners_pred, players_pred)
        q50 = earners_pred / players_pred * 100.0 if players_pred > 0 else last_rate

    if q50 is None:
        q50 = float(last_rate)

    # Do not let the median run away from the last rate without limit.
    lg_last = logit(last_rate)
    lg_q50 = logit(q50)
    max_shift = 1.2 + 0.9 * math.sqrt(max(0.0, horizon_days))
    lg_q50 = min(lg_last + max_shift, max(lg_last - max_shift, lg_q50))
    if flat_history:
        # A refresh is overdue and unseen.  When it lands, the tracked
        # player count jumps by far more than the earner counts of most
        # trophies, so rates are diluted downward; nudge the median down
        # a little rather than pretending flat means stable.
        lg_q50 -= 0.30
    q50 = inv_logit(lg_q50)

    # Spread: horizon growth, own step volatility, fit residual, staleness.
    sigma = math.sqrt((0.42 * math.sqrt(max(0.0, horizon_days))) ** 2
                      + (0.55 * volatility * math.sqrt(max(0.0, horizon_days))) ** 2
                      + (0.45 * resid) ** 2
                      + stale_sigma ** 2)
    if flat_history:
        sigma = max(sigma, 0.90)
    sigma = min(1.35, max(0.30, sigma))

    lg = logit(q50)
    qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
    qs = sorted(qs)
    return {"id": tid, "q": qs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fallback_forecast(case):
    """Persistence for every trophy; used if the main model fails outright."""
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    history = case.get("history") or []
    last_rates = []
    if history and isinstance(history[-1], dict):
        last_rates = history[-1].get("rates") or []
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) else None
        rows.append(persistence_row(trophy.get("id"), rate, horizon_days))
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = fallback_forecast(case)
            except Exception:
                forecast = {"trophies": []}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            forecast = fallback_forecast(case)
            text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
