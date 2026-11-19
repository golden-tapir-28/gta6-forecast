[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/golden-tapir-28/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/golden-tapir-28/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at 2026-11-26T00:00:00Z — one week after launch — as
five quantiles per trophy. Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The program

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads every `<id>.in` case in the input directory and writes
`<id>.out` with `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

The model, in brief:

- Per trophy, a least-squares line of logit(rate) against
  ln(1 + hours since release) over the last 48 hours of reads — log time
  is the right clock for a launch, because progress is frantic at first
  and then decelerates.
- Each trophy's slope is shrunk 30% toward its tier's median slope, so
  one trophy's noisy jump never becomes a wild forecast.
- The trend is blended with the last read in logit space, trusting the
  trend more the more distinct read times support it.
- A stale page (nothing moved for two hours or more) is treated as a
  refresh about to happen: the median is shifted down — Exophase's
  refresh dilutes every rate with newly tracked players — and the
  spread is widened. Launch night's page sat frozen at 638 players for
  hours, then refreshed to 3,868.
- Quantile spread grows with the square root of the horizon on the logit
  scale, plus the fit's residual variance — honest about a 4.5-day
  forecast, not a 90-minute one.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Try it on the practice cases in the kit's `input/` directory
(`p00.in`, `p01.in`): output is valid, deterministic across runs, and
runs in well under a second.
