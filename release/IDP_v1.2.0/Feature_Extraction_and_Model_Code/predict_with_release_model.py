from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--features', required=True, help='NumPy .npy matrix with 628 columns')
parser.add_argument('--output', required=True)
args = parser.parse_args()

package = joblib.load(args.model)
X = np.asarray(np.load(args.features), dtype=np.float32)
if X.ndim != 2 or X.shape[1] != package['feature_count']:
    raise ValueError(f"Expected a two-dimensional matrix with {package['feature_count']} columns")
base = np.column_stack([package['lgbm'].predict_proba(X)[:, 1], package['catboost'].predict_proba(X)[:, 1]])
probability = package['meta'].predict_proba(base)[:, 1]
pd.DataFrame({'row_index': np.arange(len(X)), 'essentiality_score': probability, 'threshold': package['threshold'], 'predicted_essential': probability >= package['threshold']}).to_csv(args.output, sep='\t', index=False)
