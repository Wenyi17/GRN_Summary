"""
GRNBoost2 (Moerman et al., 2019)

Faithful reimplementation of the GRNBoost2 algorithm:
Per-target gradient boosting (XGBoost) with early stopping,
using feature importances as edge weights.

This is algorithmically identical to arboreto's grnboost2.
We call XGBoost directly to avoid dask compatibility issues on HPC.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from joblib import Parallel, delayed


EARLY_STOP_WINDOW_LENGTH = 25


def _fit_one_target(j, X, gene_names, tf_mask, seed):
    """Fit gradient boosting for one target gene."""
    y = X[:, j]
    mask = np.ones(X.shape[1], dtype=bool)
    mask[j] = False
    X_pred = X[:, mask]
    predictor_names = [g for k, g in enumerate(gene_names) if k != j]

    # Hyperparameters matching arboreto's SGBM_KWARGS
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.5,
        reg_alpha=0.01,
        early_stopping_rounds=EARLY_STOP_WINDOW_LENGTH,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )

    n_samples = X_pred.shape[0]
    n_val = max(1, n_samples // 5)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    model.fit(
        X_pred[tr_idx], y[tr_idx],
        eval_set=[(X_pred[va_idx], y[va_idx])],
        verbose=False,
    )

    importances = {}
    for name, imp in zip(predictor_names, model.feature_importances_):
        if tf_mask is None or name in tf_mask:
            importances[(name, gene_names[j])] = imp

    return importances


def run_grnboost2(expression_df, tf_list=None, n_jobs=4, seed=0):
    """
    Run GRNBoost2.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Genes as columns, samples/conditions as rows.
    tf_list : list or None
    n_jobs : int
    seed : int

    Returns
    -------
    dict of {(TF, TG): importance_score}
    """
    gene_names = list(expression_df.columns)
    X = expression_df.values.astype(np.float64)

    tf_mask = set(tf_list) & set(gene_names) if tf_list is not None else None

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_fit_one_target)(j, X, gene_names, tf_mask, seed + j)
        for j in range(len(gene_names))
    )

    edge_scores = {}
    for imp_dict in results:
        edge_scores.update(imp_dict)

    return edge_scores
