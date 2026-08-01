# Predicting Smartphone Addiction

[Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8/overview). Binary classification, evaluated on ROC AUC.

## Data

| | rows | cols |
|---|---|---|
| `data/train.csv` | 691,369 | 14 |
| `data/test.csv` | 296,302 | 13 |

Target `addicted_label`, positive rate 0.709.

9 numeric features (`age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`) and 3 categorical (`gender`, `stress_level`, `academic_work_impact`).

Every feature has missing values, 4-20% per column, mean 1.26 missing per row. Missingness is uninformative: P(y=1 | col is NaN) is within 0.004 of the base rate for all 12 columns, and null rates differ between train and test, so it looks injected per split rather than carried from the source data.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install pandas scikit-learn lightgbm jupyter
```

Run `notebook.ipynb` top to bottom. It writes `submission.csv`.

## Log

Local score is ROC AUC. Public LB is the Kaggle score on ~20% of test.

| # | Change | Local | Public LB |
|---|---|---|---|
| 1 | LightGBM defaults, one-hot categoricals, 80/20 holdout | 0.95461 | 0.95556 |

### 1. Baseline

`LGBMClassifier(random_state=42)` at stock settings (100 trees, lr 0.1) in a sklearn pipeline. `OneHotEncoder(handle_unknown='ignore')` on the 3 categoricals, numerics passed through untouched. NaNs left for LightGBM to handle natively. Validated on a single contiguous 80/20 split; the model that produced `submission.csv` was fit on that 80% only.

Gain-based importance is dominated by behavioural volume: `app_opens_per_day` (743) and `notifications_per_day` (613), then `daily_screen_time_hours` (389), `social_media_hours` (358), `weekend_screen_time` (354). The categoricals are near-zero (max 4), so one-hot encoding is buying nothing.

Null result: missing-value indicators are not worth building, per the missingness check above.
