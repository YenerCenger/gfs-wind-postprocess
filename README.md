# gfs-wind-postprocess

Correcting day-ahead **GFS wind forecasts** for İzmir with machine learning.

Global weather models like GFS run on a ~13–25 km grid, too coarse to see local effects such as the summer sea breeze (*imbat*) or terrain channelling around İzmir. At İzmir Adnan Menderes Airport (LTBJ) the day-ahead GFS forecast **underestimates wind speed by about 3.7 knots on average**, and the error grows with wind strength and in summer. This project learns that error from past forecasts and observations and corrects it. This kind of correction is called *statistical post-processing* in meteorology.

> This is also a learning project: it is built step by step, from a constant bias correction up to an LSTM, to understand each model rather than just chase a score. Notebooks and code comments are in Turkish.

## Data

| Source | What | Notes |
|---|---|---|
| [Iowa Environmental Mesonet (IEM)](https://mesonet.agron.iastate.edu/request/download.phtml) | LTBJ METAR observations: wind speed, direction, gust | ~2 reports per hour, 10-minute mean wind in knots |
| [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | GFS forecast made ~24 h before each target hour (`*_previous_day1`) | A true day-ahead forecast; available from March 2024 |

Period: 2024-03-01 → 2026-11-02, hourly, UTC (~23,400 aligned hours). Each GFS hour is matched to the nearest METAR report within ±15 minutes.

Data is **not** stored in this repository; it is downloaded by a script (see below).

## Results

Time-based split: train before 2025-11-01, test from 2025-11-01 to 2026-11-02 (one full year, every season). Inputs are only forecast values and time features. Observed values are never used as inputs, to avoid leakage. Metric: mean absolute error (MAE) of wind speed in knots, lower is better.

| Model | Test MAE (kn) |
|---|---|
| Raw GFS | 4.41 |
| Constant bias correction | 3.80 |
| Monthly bias correction | 3.65 |
| Linear regression + hand-made features (hour, month, wind direction u/v, gust, northerly flag) | 2.90 |
| Gradient boosting (`HistGradientBoostingRegressor`, 125 trees) | **2.42** |
| MLP, PyTorch (2×64, mean of 3 seeds) | 2.46 |
| LSTM over the previous 12 GFS hours (mean of 3 seeds) | **2.41** |

Gradient boosting and the LSTM are effectively tied; the gap is within seed-to-seed noise.

What the data shows:
- GFS almost never overestimates: the error is one-sided, which is why a model can learn to correct it.
- The largest errors are in **summer** (mean absolute error 5.7 kn vs 3.6 kn in winter) and in **strong wind** (≈10.6 kn when the observed wind is 20 kn or more).
- The error depends heavily on **direction**: GFS is ~1.7–1.8× too low in northerly flow and close to correct in southerly flow.
- In light wind (< 10 kn) GFS is both nearly unbiased and consistent.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YenerCenger/gfs-wind-postprocess.git
cd gfs-wind-postprocess
uv sync
uv run python -m ruzgar.download   # writes data/raw/*.parquet
uv run jupyter lab
```

The Python package is named `ruzgar` (Turkish for "wind").

## Repository layout

```
src/ruzgar/
  download.py   download METAR and GFS data
  data.py       align forecasts with observations, time split
notebooks/
  01-veri          data, alignment, baseline and bias corrections
  02a-linreg       linear regression with hand-made features
  02b-boosting     gradient boosting
  03a-mlp-elle     MLP written by hand (manual weights and gradient descent)
  03b-mlp-torch    MLP with nn.Module and Adam
  03c-lstm         LSTM over a window of GFS hours
  b1…b3            pandas practice on this data (filtering, new columns, groupby)
```

## Roadmap

- [ ] Separate metric for strong wind (≥ 15 kn), the case that matters most for windsurfing
- [ ] Combine the boosting and LSTM models
- [ ] More stations (Çeşme / Alaçatı)
- [ ] Simple website with the corrected daily forecast

## Credits

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0) and the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/).
