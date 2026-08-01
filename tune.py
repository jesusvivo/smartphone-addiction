"""Optuna search for LightGBM params. Run manually, not part of the notebook."""

import json
import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

CAT = ['gender', 'stress_level', 'academic_work_impact']
TARGET = 'addicted_label'
SEARCH_LR = 0.05
N_TRIALS = 40
BASELINE_OOF = 0.95504

train = pd.read_csv('data/train.csv').drop(columns=['id'])
y = train[TARGET].values
X = train.drop(columns=TARGET)

pre = ColumnTransformer(
    [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT)],
    remainder='passthrough')
Xe = pre.fit_transform(X)

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xe, y))
tri, vai = folds[0]

FIXED = dict(objective='binary', random_state=42, verbose=-1, n_jobs=-1, subsample_freq=1)


def objective(trial):
    params = dict(
        num_leaves=trial.suggest_int('num_leaves', 31, 512, log=True),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 300, log=True),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        max_bin=trial.suggest_int('max_bin', 127, 511),
    )
    m = LGBMClassifier(**FIXED, learning_rate=SEARCH_LR, n_estimators=3000, **params)
    m.fit(Xe[tri], y[tri], eval_set=[(Xe[vai], y[vai])], eval_metric='auc',
          callbacks=[early_stopping(100, verbose=False)])
    trial.set_user_attr('best_iter', int(m.best_iteration_))
    return roc_auc_score(y[vai], m.predict_proba(Xe[vai])[:, 1])


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS)

bp = study.best_params
bi = study.best_trial.user_attrs['best_iter']
print(f'search done in {time.time() - t0:.0f}s')
print(f'best fold-0 AUC {study.best_value:.5f} at {bi} trees')
print(json.dumps(bp, indent=2))

# n_estimators pinned so folds 1-4 are untouched by the search
print(f'\nverifying: 5-fold, n_estimators={bi}, no early stopping')
oof = np.zeros(len(y))
aucs = []
for k, (tr, va) in enumerate(folds):
    m = LGBMClassifier(**FIXED, learning_rate=SEARCH_LR, n_estimators=bi, **bp).fit(Xe[tr], y[tr])
    oof[va] = m.predict_proba(Xe[va])[:, 1]
    aucs.append(roc_auc_score(y[va], oof[va]))
    print(f'  fold {k} {aucs[-1]:.5f}')

pooled = roc_auc_score(y, oof)
print(f'\npooled OOF     {pooled:.5f}   (baseline {BASELINE_OOF}, delta {pooled - BASELINE_OOF:+.5f})')
print(f'folds 1-4 mean {np.mean(aucs[1:]):.5f}')
print(f'fold std       {np.std(aucs):.5f}')

json.dump({'params': bp, 'n_estimators': bi, 'learning_rate': SEARCH_LR,
           'pooled_oof': pooled}, open('best_params.json', 'w'), indent=2)
