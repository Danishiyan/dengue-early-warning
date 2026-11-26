# Product requirements: Sri Lanka dengue early warning

*Written on 26 Nov 2026, after the first version of the model and API was built, to record the goals, scope and limits of v1 and plan v2.*

## 1. Problem
Dengue is reported weekly for 26 health areas in Sri Lanka (25 districts + Kalmunai). Outbreaks rise quickly, and the weekly reports only show what already happened. A simple forecast of **next week's cases per area** can give health teams a few extra days to plan mosquito control, hospital beds and public warnings.

## 2. Users
| User | What they need | What they do with a forecast |
|---|---|---|
| Regional health officer (e.g. a Medical Officer of Health team) | Next week's expected cases for their area | Send fogging and breeding-site inspection teams early to areas where cases are expected to rise |
| District hospital planner | An early signal of rising cases | Prepare beds, staff and test kits before a peak |
| Data analyst or researcher | A simple, documented API and baseline results | Compare areas, build dashboards, or test better models |

## 3. Goal and success metrics
| Goal | Metric | v1 status |
|---|---|---|
| Forecast better than the simplest guess | Average MAE below the "same as last week" baseline in time-series CV (2021–2025) | ✅ 10.7 vs 11.3 cases per area per week |
| Useful in most areas | Model beats the baseline in most areas in the latest full year | ✅ 23 of 26 areas (2025) |
| Easy to use | Public API with `/health`, `/areas`, `/forecast/{area}` and docs | ✅ locally and in Docker; ☐ online |
| Safe to run | Cloud deployment with a budget alert and basic monitoring | ☐ planned |

## 4. Scope (v1)
- Weekly case data from 2010 onwards for 26 health areas.
- Features: this week's cases, lags 1–8, 4-week rolling mean, week of year, area.
- Gradient boosting model, compared with naive baselines using time-series cross-validation.
- FastAPI service in a Docker image, tested in GitHub Actions CI.

## 5. Non-goals (not in v1)
- Weather data (rainfall, temperature): planned for **v2**.
- Daily or village-level forecasts.
- A public website or mobile app for the general public.
- An official warning system: forecasts support people's decisions; they do not replace them.

## 6. Risks and limits
| Risk | Effect | Mitigation |
|---|---|---|
| Model is worse than the baseline in Colombo and Gampaha, the two biggest areas | Weaker warnings where most cases happen | State it clearly in the model card; v2 adds weather and tries predicting % change |
| The latest weeks are often revised as reports arrive | Forecasts from incomplete data | Last 4 weeks flagged as `is_recent`; the API forecasts from the latest fully reported week and returns `weeks_still_being_reported` |
| Missing or irregular weeks in the source data | Wrong lag values | Missing weeks are added as unknown and their rows dropped (with a warning and a test) |
| Free cloud tiers (memory limits, sleeping services) | Slow first request or crashes | Small image, `/health` check, document cold starts |
| Case history alone can't foresee sudden outbreaks | Late warnings for sharp spikes | v2: weather features; communicate uncertainty |

## 7. Plan
| Version | Adds |
|---|---|
| v1.0.0 | Current model + API, deployed with a budget alert and monitoring, model card |
| v2 | Weather features per area, experiment tracking, re-evaluation against v1 |
