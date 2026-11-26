# Model card: next-week dengue cases per health area (v1)

*Written on 26 Nov 2026 for v1. Figures come from `reports/` and are reproduced by `python -m src.train` and `python -m src.evaluate`.*

## Model details
| Item | Value |
|---|---|
| Task | Regression: predict next week's reported dengue cases for one health area |
| Algorithm | scikit-learn `HistGradientBoostingRegressor` |
| Loss | `absolute_error` (matches the reported metric, MAE) |
| Inputs (13) | this week's cases, `lag_1` … `lag_8`, `rolling_mean_4`, `week_sin`, `week_cos`, `district` (categorical) |
| Output | Expected cases next week (clipped at 0 in the API) |
| Seed | `random_state = 42` |
| Settings | `configs/config.yaml`; runs tracked with MLflow (`src/experiment.py`) |

## Intended use
- Give regional health teams and analysts an early signal of next week's dengue cases per area, to help plan mosquito control and hospital capacity.
- Compare areas and weeks alongside other information, not on its own.

## Out of scope
- Official outbreak declarations or any decision made without human review.
- Village-level or daily forecasts.
- Forecasts more than one week ahead.
- Use outside Sri Lanka or with data in a different format.

## Data
| Item | Value |
|---|---|
| Source | Ministry of Health weekly dengue cases via the `denguedatahub` R package (Thiyanga Talagala, GPL-3) |
| Coverage | 26 health areas (25 districts + Kalmunai), weekly, from 2010 (earlier years left out after EDA) |
| Rows used | 22,429 area-weeks with complete lag features (27 Feb 2010 – 7 Sep 2026) |
| Known issues | Latest 4 weeks are often revised (flagged `is_recent`, not used as targets); one missing week (Puttalam, 9 Feb 2026) is marked unknown and the rows that depend on it are dropped; the source's week start changed at New Year 2026 |

## Evaluation
- **Method:** time-series cross-validation. For each test year 2021–2025, train only on earlier years and predict every week of the test year, so the model never sees the future.
- **Metric:** MAE, the average size of the error in cases per area per week (lower is better).
- **Baseline:** "same as last week" (next week = this week), the strongest of three naive baselines.

### Results by test year
| Test year | Baseline MAE | Model MAE | Better? |
|---|---|---|---|
| 2021 | 7.1 | 6.8 | ✅ |
| 2022 | 12.1 | 10.9 | ✅ |
| 2023 | 18.0 | 18.1 | ❌ (largest outbreak year) |
| 2024 | 10.7 | 9.5 | ✅ |
| 2025 | 8.8 | 8.2 | ✅ |
| **Average** | **11.3** | **10.7** | about 5–6% lower error |

The exact gain is 5.6% on macOS and 6.0% in the Linux Docker/CI build; the averages are the same.

### Results by area (2025)
The model beats the baseline in **23 of 26** areas. It is worse in three:

| Area | Cases in 2025 | Baseline MAE | Model MAE |
|---|---|---|---|
| Colombo | 12,072 | 35.3 | 36.8 |
| Gampaha | 7,711 | 20.3 | 23.1 |
| Trincomalee | 1,064 | 5.2 | 5.4 |

Full table: `reports/error_by_area_2025.csv`.

## Fairness and risk
- **Uneven quality across areas.** The model is weakest in Colombo and Gampaha, which together had about 39% of 2025's cases. Relying on it there could mean weaker early warnings where the most people are affected. Users in these areas should rely on the baseline or other signals until v2 improves them.
- **Outbreak years.** In 2023, the biggest outbreak year, the model was no better than the baseline. Case history alone cannot anticipate sudden spikes.
- **Reporting quality.** Areas with late or missing reports get less reliable forecasts; the API returns `data_may_be_incomplete` for recent weeks.
- No personal data is used; all data is aggregated counts per area and week.

## Limitations and next steps (v2)
- No weather inputs yet; rainfall and temperature lead dengue by a few weeks and are planned for v2.
- Try predicting the % change for large areas, where absolute errors are biggest.
- Re-evaluate against these v1 numbers in MLflow (runs tagged `version = v1`).

## Versioning
| Version | Date | Change |
|---|---|---|
| v1 | Nov 2026 | First model card: case-history features only |
