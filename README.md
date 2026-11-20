# dengue-early-warning

Forecasting weekly dengue risk for Sri Lankan districts from case and weather data.

> 🚧 Work in progress — Project 2 of my AI/ML portfolio.

## Problem
Dengue is one of Sri Lanka's biggest public-health problems. Outbreaks follow rain and
temperature with a delay of a few weeks. This project predicts **next weeks' dengue cases
per district** so that health teams could act early.

## Planned approach
1. ✅ Collect weekly dengue cases per district (see **Data** below)
2. Join weekly weather per district (rainfall, temperature, humidity)
3. Build time-series features (lags, rolling averages)
4. Compare against simple baselines, then train ML models with time-series cross-validation
5. Serve predictions with an API, package with Docker, deploy to Azure

## Data
| | |
|---|---|
| **Source** | Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka — in tidy form via the [denguedatahub](https://github.com/thiyangt/denguedatahub) R package by Thiyanga Talagala (GPL-3) |
| **Coverage** | Week 52 of 2006 → week 37 of 2026 (weekly) |
| **Areas** | 26 health areas: the 25 districts + Kalmunai (a separate regional health area) |
| **Size** | 26,753 rows · columns `year, week, start_date, end_date, district, cases` |
| **Quality checks** | 0 missing values · 0 duplicate (year, week, area) rows · two date formats in the source, unified while loading |

The data is **not stored in this repo**. One command downloads it and converts it to CSV:

```bash
pip install -r requirements.txt
python -m src.download_data
# -> data/raw/srilanka_weekly_data.rda (original)
# -> data/raw/dengue_weekly.csv       (tidy CSV)
```

## Project structure
```
data/
  raw/          # original downloaded data (not in Git)
  external/     # extra sources, e.g. weather (not in Git)
  processed/    # cleaned, joined data (not in Git)
notebooks/      # exploration
src/            # project code
tests/          # automated tests
configs/        # settings (paths, districts, parameters)
models/         # trained models (not in Git)
reports/figures # charts
```

## Licence
MIT
