"""Train one deployment model from a public all_labeled_features.npz file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def base_lgbm(seed: int):
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, subsample=0.85,
        colsample_bytree=0.60, reg_alpha=0.1, reg_lambda=5,
        class_weight="balanced", random_state=seed, n_jobs=8, verbosity=-1,
    )


def base_catboost(seed: int):
    return CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.04, loss_function="Logloss",
        l2_leaf_reg=8, random_seed=seed, verbose=False, thread_count=4,
        allow_writing_files=False,
    )


def mcc_threshold(y, probability):
    best_threshold, best_mcc = 0.5, -2.0
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = probability >= threshold
        tp = np.sum((prediction == 1) & (y == 1)); tn = np.sum((prediction == 0) & (y == 0))
        fp = np.sum((prediction == 1) & (y == 0)); fn = np.sum((prediction == 0) & (y == 1))
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = (tp * tn - fp * fn) / denominator if denominator else -1.0
        if mcc > best_mcc:
            best_threshold, best_mcc = float(threshold), float(mcc)
    return best_threshold


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="all_labeled_features.npz")
parser.add_argument("--species", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

data = np.load(args.data)
X, y = np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int8)
feature_names = data["feature_names"].astype(str).tolist()
oof = np.zeros((len(y), 2), dtype=np.float32)
for fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=20260813).split(X, y)):
    lgb_model = base_lgbm(100 + fold).fit(X[train_index], y[train_index])
    cat_model = base_catboost(200 + fold).fit(X[train_index], y[train_index])
    oof[test_index, 0] = lgb_model.predict_proba(X[test_index])[:, 1]
    oof[test_index, 1] = cat_model.predict_proba(X[test_index])[:, 1]
meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=20260813).fit(oof, y)
package = {
    "species": args.species,
    "model_type": "two-base-learner OOF-stacked deployment ensemble",
    "feature_count": int(X.shape[1]), "feature_names": feature_names,
    "threshold": mcc_threshold(y, meta.predict_proba(oof)[:, 1]),
    "threshold_source": "3-fold out-of-fold MCC maximization",
    "lgbm": base_lgbm(20260813).fit(X, y),
    "catboost": base_catboost(20260813).fit(X, y), "meta": meta,
}
output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(package, output, compress=3)
(output.with_suffix(".json")).write_text(json.dumps({k: v for k, v in package.items() if k not in {"lgbm", "catboost", "meta"}}, indent=2), encoding="utf-8")
