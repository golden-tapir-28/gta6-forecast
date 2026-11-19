"""Forecast GTA VI trophy earn rates on Exophase a week after launch.

Entry for make.fun gta6-trophy-rarity-forecast-01. Python 3.13, standard
library only, no network, deterministic: the same case always yields the
same bytes.

The model
---------
A trophy's earn rate is the share of Exophase-tracked players who hold it.
Over launch week that share moves for two reasons that pull in opposite
directions: players earn trophies as they play (rates rise, fast for early
story trophies, slow for grinds, barely at all for the rarest), and
Exophase keeps tracking new players who own the game but have earned
nothing yet (each refresh dilutes every rate). The house baselines ignore
both: the flat baseline repeats the last rate, the curve baseline fits a
straight line to the logit rate against log time. This program instead:

1. Fits, per trophy, a least-squares line of logit(rate) against
   ln(1 + hours since release) over the last 48 hours of reads. Log time
   is the right clock for a launch: progress is frantic in the first hours
   and decelerates, so a line in log time extrapolates a saturating curve,
   not a runaway straight line.
2. Shrinks each trophy's slope 30% toward the median slope of its tier.
   Single-trophy slopes are noisy (rates are printed to 2 decimals, and
   rare trophies move by one earner at a time), and shrinkage is what
   keeps a lucky jump in one trophy from becoming a wild forecast.
3. Blends the trend forecast with the last read in logit space, trusting
   the trend more the more distinct read times the case holds.
4. Detects a stale page: when every rate and the players-tracked count
   have been frozen for two hours or more, Exophase has not refreshed and
   the next refresh will dilute the rates with a wave of new players.
   The median is shifted down (about 1.1 logit units on launch night,
   decaying as the launch ages) and the spread widened, because the
   practice cases show exactly this: launch night's page sat unchanged
   at 638 players for hours, then refreshed to 3,868, and the house
   baseline's practice loss of ~1.51 says the rates fell to about a
   third of the frozen page's.
5. Spreads the five quantiles on the logit scale with a sigma that grows
   with the square root of the horizon (a random walk in logit space),
   plus the fit's residual variance, plus the staleness term. A spread
   that is honest about a 4.5-day horizon: too narrow pays on every
   trophy the rate escapes, too wide pays on every trophy.

Quantiles use the standard normal z-scores at 0.05/0.25/0.5/0.75/0.95,
medians are capped by tier (a platinum cannot plausibly sit above 12% a
week in, a gold above 45%), and every output is rounded to 4 decimals,
half up, exactly like the starter's rounding, so runs are byte-identical.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
It reads every <id>.in in the input directory ($MAKEFUN_INPUT_DIR, else
/input) and writes <id>.out to the output directory ($MAKEFUN_OUTPUT_DIR,
else /output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry ships none.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# How fast the logit spread widens: sigma per square-root day of horizon.
SIGMA_PER_SQRT_DAY = 0.35
# How far the median is pulled down when the page is stale (a refresh is
# due and will dilute every rate with newly tracked players). Calibrated
# on the practice cases: the house baseline's stated practice loss of
# ~1.51 pins the refresh's earner growth at only ~1.3x while the tracked
# base jumped 638 -> 3,868, i.e. rates fell to roughly a third of the
# frozen page's -- about 1.1 logit units down. The shift decays with the
# age of the launch: early on a refresh multiplies a tiny tracked base,
# later the same refresh adds proportionally far fewer new players.
STALE_SHIFT_LOGIT = 1.1
STALE_SIGMA_LOGIT = 0.5  # extra spread (logit) while a refresh is due.
# A page counts as stale when nothing on it moved for this long.
STALE_MINUTES = 120.0
# The fit window: reads newer than this before the checkpoint.
FIT_WINDOW_HOURS = 48.0
# Slope shrinkage toward the tier median: keep 70% of a trophy's own slope.
OWN_SLOPE_WEIGHT = 0.70
# Plausible week-one ceilings by tier, in percent. These are sanity caps,
# not targets: a median forecast is clamped into [0.02, cap].
TIER_CAP = {"platinum": 12.0, "gold": 45.0, "silver": 75.0, "bronze": 99.0}
# Fallback median (percent) when a trophy has no usable read at all.
TIER_PRIOR = {"platinum": 0.3, "gold": 0.6, "silver": 0.9, "bronze": 1.0}


def parse_time(value: str) -> datetime:
    """Parse an ISO 8601 timestamp (or bare date) as aware UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, as the starter does.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b*x. Returns (intercept, slope, residual var)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n
    return intercept, slope, resid


def page_is_stale(history: list[dict], checkpoint: datetime) -> bool:
    """True when the page has not moved at all for STALE_MINUTES or more.

    Exophase refreshes on its own schedule; a page frozen for hours while
    a game is freshly launched is a page about to be refreshed, and the
    refresh dilutes every rate. All reads identical over a long span is
    the fingerprint: launch night sat at 638 players, every rate the
    same, from 23:48 until 04:00 UTC.
    """
    if len(history) < 2:
        return False
    # Look back well beyond the staleness threshold, so a page frozen for
    # the whole threshold can actually show a span that long.
    window_start = checkpoint.timestamp() - 3 * STALE_MINUTES * 60
    recent = [r for r in history
              if parse_time(r["at"]).timestamp() >= window_start]
    if len(recent) < 2:
        return False
    first = recent[0]
    span = (parse_time(recent[-1]["at"]) - parse_time(recent[0]["at"])
            ).total_seconds() / 60.0
    if span < STALE_MINUTES:
        return False
    for read in recent[1:]:
        if read["players_tracked"] != first["players_tracked"]:
            return False
        if read["rates"] != first["rates"]:
            return False
    return True


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    release = parse_time(case["game"]["release_date"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def log_clock(moment: datetime) -> float:
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    x_target = log_clock(target)
    stale = page_is_stale(history, checkpoint)
    # A refresh due soon after launch dilutes rates far more than one due
    # days in, when the tracked base is already large.
    stale_factor = 1.0 / (1.0 + (checkpoint - release).total_seconds()
                          / 86400.0)
    stale_shift = STALE_SHIFT_LOGIT * stale_factor
    stale_var = (STALE_SIGMA_LOGIT * stale_factor) ** 2

    # The fit window: the last 48 hours of reads, or all of them when the
    # case holds fewer than three distinct read times.
    cutoff = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600
    window = [r for r in history
              if parse_time(r["at"]).timestamp() >= cutoff]
    if len({r["at"] for r in window}) < 3:
        window = list(history)
    window_xs = [log_clock(parse_time(r["at"])) for r in window]

    # First pass: each trophy's own trend line, to get the tier medians
    # the slopes shrink toward.
    own: list[dict] = []
    for i, trophy in enumerate(trophies):
        points = [(x, logit(r["rates"][i])) for x, r in zip(window_xs, window)
                  if r["rates"][i] is not None]
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if len(points) >= 2 and len({x for x, _ in points}) >= 2:
            intercept, slope, resid = fit_line([p[0] for p in points],
                                               [p[1] for p in points])
            distinct = len({p[0] for p in points})
        else:
            intercept = logit(last_rate) if last_rate is not None else logit(
                TIER_PRIOR.get(trophy["tier"], 1.0))
            slope, resid, distinct = 0.0, 0.0, 0
        own.append({"trophy": trophy, "intercept": intercept, "slope": slope,
                    "resid": resid, "distinct": distinct,
                    "last_rate": last_rate})

    tier_slopes: dict[str, list[float]] = {}
    for entry in own:
        tier_slopes.setdefault(entry["trophy"]["tier"], []).append(
            entry["slope"])
    tier_median = {t: median_of(s) for t, s in tier_slopes.items()}

    rows = []
    for entry in own:
        trophy = entry["trophy"]
        tier = trophy["tier"]
        # Shrink the slope toward the tier median: one trophy's noisy
        # jumps do not become a wild forecast.
        slope = (OWN_SLOPE_WEIGHT * entry["slope"]
                 + (1.0 - OWN_SLOPE_WEIGHT) * tier_median.get(tier, 0.0))
        trend_logit = entry["intercept"] + slope * x_target
        if entry["last_rate"] is not None:
            last_logit = logit(entry["last_rate"])
            # Trust the trend more the more distinct read times support it.
            weight = min(1.0, entry["distinct"] / 6.0)
            median_logit = (weight * trend_logit
                            + (1.0 - weight) * last_logit)
        else:
            median_logit = trend_logit

        # A trophy everyone already holds stays held; a stale page's
        # coming refresh dilutes everything else.
        saturated = (entry["last_rate"] is not None
                     and entry["last_rate"] >= 99.99)
        if stale and not saturated:
            median_logit -= stale_shift

        # A trophy everyone already holds is not capped down by its tier:
        # 100% stays (essentially) 100%.
        cap = 99.99 if saturated else TIER_CAP.get(tier, 99.0)
        median_pct = min(cap, max(0.02, sigmoid(median_logit)))

        # Spread: a random walk in logit space over the horizon, plus the
        # fit's residual variance, plus the staleness term. Wider when the
        # forecast reaches further and when the data behind it is noisier.
        var = ((SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
               + min(entry["resid"], 0.25)
               + (stale_var if stale and not saturated else 0.0))
        spread = min(1.5, max(0.18, math.sqrt(var)))
        if saturated:
            spread = 0.05

        centre = logit(median_pct)
        quantiles: list[float | int] = []
        previous = -1.0
        for z in Z:
            value = round4(sigmoid(centre + z * spread))
            value = min(100, max(0, value))
            # Rounding can only break monotonicity by a hair; enforce it.
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = float(value)
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
