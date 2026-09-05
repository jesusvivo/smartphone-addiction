# Predicting Smartphone Addiction

[Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8/overview). Binary classification, evaluated on ROC AUC.

## Result

Final private leaderboard **0.96643**, placing **1320 of 3532** (top 37%).

Public 0.96668, private 0.96643, a drop of 0.00025. Essentially no shakeup, which is what you would expect here: the model was never selected against the public leaderboard. Every keep/revert decision came from 5-fold OOF, and the leaderboard was only ever used to confirm a decision already made.

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
python -m venv .venv && .venv/bin/pip install pandas scikit-learn lightgbm optuna jupyter
```

Run `notebook.ipynb` top to bottom. It writes `submission.csv`.

`tune.py` is a standalone Optuna search, kept out of the notebook so a clean run-all stays cheap. Run it only when re-tuning; it writes `best_params.json`, which is transcribed into the notebook's `PARAMS`.

## Validation

Measured 5-fold `StratifiedKFold` fold-to-fold std at 0.00065 ROC AUC. Public LB is scored on ~20% of test (~59k rows), which puts its standard error near 0.0008.

Decision rule: keep a change if it gains more than 0.0015 on 5-fold CV mean, revert if it loses more than 0.0015, and treat anything in between as a null result. A single 80/20 holdout cannot resolve differences at this scale and should not be used to accept or reject a change.

## Log

CV is 5-fold `StratifiedKFold` pooled out-of-fold ROC AUC. Public LB is the Kaggle score. Iterations 1-3 predate the CV harness and were judged on a single 80/20 holdout, which is why their CV figures are backfilled.

| # | Change | CV | Public LB | Verdict |
|---|---|---|---|---|
| 1 | LightGBM defaults, one-hot categoricals | 0.95494 ± 0.00065 | 0.95556 | baseline |
| 2 | Ordinal encoding instead of one-hot | 0.95505 ± 0.00058 | 0.95588 | null, kept for simplicity |
| 3 | Refit on 100% of train before predicting | | 0.95588 | void, never actually ran |
| 4 | Refit on 100% of train, for real | | 0.95584 | kept on principle, LB delta is noise |
| 5 | 5-fold CV harness in the notebook | | | no model change |
| 6 | Optuna hyperparameter search, 40 trials | 0.96532 | **0.96668** | **kept, +0.01084 on LB** |
| 7 | `is_unbalance=True` on tuned params | 0.96516 | | null, reverted |
| 8 | lr 0.02 with 4800 trees | 0.96553 | | null, reverted |

Final submission was iteration 6: public 0.96668, private **0.96643**.

### 1. Baseline

`LGBMClassifier(random_state=42)` at stock settings (100 trees, lr 0.1) in a sklearn pipeline. `OneHotEncoder(handle_unknown='ignore')` on the 3 categoricals, numerics passed through untouched. NaNs left for LightGBM to handle natively. Validated on a single contiguous 80/20 split; the model that produced `submission.csv` was fit on that 80% only.

Gain-based importance is dominated by behavioural volume: `app_opens_per_day` (743) and `notifications_per_day` (613), then `daily_screen_time_hours` (389), `social_media_hours` (358), `weekend_screen_time` (354). The categoricals are near-zero (max 4), so one-hot encoding is buying nothing.

Null result: missing-value indicators are not worth building, per the missingness check above.

### 2. Ordinal encoding

Swapped `OneHotEncoder` for `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`. Everything else unchanged.

The holdout moved +0.00033 and the LB +0.00032, both well inside noise. Ran 5-fold CV across three encodings to settle it:

| Encoding | CV | vs one-hot |
|---|---|---|
| one-hot | 0.95494 ± 0.00065 | |
| ordinal | 0.95505 ± 0.00058 | +0.00010, wins 3/5 folds |
| native `category` dtype | 0.95495 ± 0.00064 | +0.00000, wins 1/5 folds |

All three are the same model. Native categorical is near bit-identical to one-hot (per-fold deltas of 1e-5 or exactly 0), which follows from the features having only 2-3 levels each, where a categorical split and the one-hot splits express the same partitions. The three categoricals together are ~0.2% of total gain importance, so encoding is not a lever on this dataset.

Kept ordinal because it produces 12 columns instead of 20, not because it scored higher.

### 3. Full-data refit (void)

Removed the train/validation split so the model would fit on all 691,369 rows. It never happened, and the submission was a duplicate of iteration 2.

The split cell stopped defining `X_train`/`y_train`, but the fit cell still called `pipeline.fit(X_train, y_train)` and silently picked up stale variables from the kernel. LightGBM logged `Number of data points in the train set: 553095`, still 80%. The scoring cell had been changed to `predict_proba(X)` against `y`, so the 0.95574 it reported was an in-sample score on data the model had trained on, not an improvement.

Verified after the fact: `submission.csv` was bit-identical to the 80%-fit predictions (max abs diff 1.1e-16), which is why the LB repeated 0.95588 exactly. A genuine full refit does move the test predictions (max abs diff 0.42, Spearman 0.998 against the 80% model), so the change is still worth making.

Honest 5-fold OOF AUC for this configuration is **0.95504**. Two lessons: restart-and-run-all before trusting a notebook number, and never evaluate on rows the model was fit on.

### 4. Full-data refit

The fit cell now reads `pipeline.fit(X, y)` and LightGBM confirms `Number of data points in the train set: 691369`.

LB went 0.95588 to 0.95584, a change of -0.00004, roughly 0.05 standard errors. Kept anyway: 25% more training data is a principled improvement, CV cannot measure it because CV evaluates the modelling procedure rather than the final fit, and the LB reading carries no information at this magnitude.

The notebook still reported an in-sample number here (`predict_proba(X)` after fitting on all of `X`), which landed at 0.95584 and looked reassuringly close to the LB. That agreement is a coincidence of untuned LightGBM at 100 trees barely overfitting, and it will disappear as soon as trees get deeper or more numerous. Replaced in the next iteration.

### 5. CV harness

Added a 5-fold `StratifiedKFold` cell that collects out-of-fold predictions and reports OOF AUC plus fold-to-fold std, placed before the final full-data fit. Removed the in-sample scoring cell.

Each fold gets `clone(pipeline)`, so the `OrdinalEncoder` is refit inside the fold and preprocessing does not leak across the split. No model change, so no submission.

### 6. Hyperparameter search

40-trial Optuna TPE search in [tune.py](tune.py), scored on fold 0 only. `learning_rate` fixed at 0.05 and `n_estimators` at 3000 with early stopping, so tree count is discovered per trial rather than searched against a correlated learning rate. Searched `num_leaves`, `min_child_samples`, `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`, `max_bin`. Search took 2062s.

Winner at 1929 trees, fold-0 AUC 0.96455:

```
num_leaves=80  min_child_samples=97  colsample_bytree=0.524
subsample=0.911  reg_alpha=1.86  reg_lambda=1.66e-05  max_bin=509
```

Verified by re-running all 5 folds with `n_estimators` pinned at 1929 and early stopping off, so folds 1-4 are untouched by the search:

| fold | 0 (search) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| AUC | 0.96455 | 0.96533 | 0.96555 | 0.96610 | 0.96509 |

Pooled OOF **0.96532** against the baseline's 0.95504, so **+0.01028**, roughly 20 fold-sigmas at a fold std of 0.00051.

No sign of search overfitting: fold 0 is the *lowest* of the five, and the honest folds-1-to-4 mean of 0.96552 sits above the pooled number. If the search had been fitting noise in its own fold, fold 0 would be the highest.

Most of the gain is simply training long enough. The very first trial, before TPE had learned anything, already scored 0.95994 on a fold where the baseline scored 0.9544. The baseline's 100 trees at lr 0.1 were badly undertrained on 691k rows; the search refined on top of that but did not create the bulk of it.

Confirmed on the leaderboard at **0.96668**, up 0.01084 from the baseline's 0.95584 against a predicted +0.01028 from CV. The prediction was registered before submitting: the baseline's LB sat 0.0008 above its OOF, so the tuned model was expected near 0.9660, and it came in at 0.96668. CV and LB now track each other closely enough to make decisions on CV alone.

### 7. Class weighting (null)

The target is 490,474 positive to 200,895 negative, so `is_unbalance=True` was worth one run on top of the tuned params. Pooled OOF 0.96516 against 0.96532, a change of -0.00016 at a fold std of 0.00051. Reverted.

Expected, for three reasons. ROC AUC depends only on the ranking of scores, and reweighting mostly rescales them. LightGBM already puts the base rate in the intercept, visible in the training log as `pavg=0.709424 -> initscore=0.892590`. And 2.4:1 is mild enough that both classes carry plenty of gradient. Reweighting earns its keep on threshold metrics like F1, or at ratios closer to 99:1.

### 8. Lower learning rate (null)

The search fixed `learning_rate` at 0.05, so dropping to 0.02 with the tree count scaled up to 4800 was worth one run. Pooled OOF 0.96553 against 0.96532, so +0.00021 for roughly 2.5x the training time. Below the 0.0015 threshold by a factor of seven, and well inside the 0.00052 fold std. Kept lr 0.05.

This is the usual shape of learning-rate tuning once a model is already trained to convergence: the first move from 0.1 to 0.05 with enough trees was worth 0.010, and the next halving was worth 0.0002.


