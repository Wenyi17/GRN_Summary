"""
TIGRESS (Haury et al., 2012)

Trustful Inference of Gene REgulation using Stability Selection.
For each target gene, runs LARS (Least Angle Regression) on many bootstrap
resamples and counts the selection frequency of each predictor gene.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lars
from joblib import Parallel, delayed


def _stability_selection_one_target(j, X, gene_names, n_bootstrap=1000,
                                     sample_fraction=0.8, seed=0):
    """Run stability selection for one target gene."""
    n_samples, n_genes = X.shape
    y = X[:, j]
    mask = np.ones(n_genes, dtype=bool)
    mask[j] = False
    X_pred = X[:, mask]
    predictor_names = [g for k, g in enumerate(gene_names) if k != j]
    n_predictors = len(predictor_names)

    selection_count = np.zeros(n_predictors)
    rng = np.random.RandomState(seed)

    subsample_size = max(2, int(n_samples * sample_fraction))

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=subsample_size, replace=True)
        X_b = X_pred[idx]
        y_b = y[idx]

        X_std = (X_b - X_b.mean(axis=0)) / (X_b.std(axis=0) + 1e-10)
        y_std = (y_b - y_b.mean()) / (y_b.std() + 1e-10)

        try:
            lars = Lars(n_nonzero_coefs=min(10, n_predictors), fit_intercept=False)
            lars.fit(X_std, y_std)
            active = np.where(np.abs(lars.coef_) > 1e-10)[0]
            selection_count[active] += 1
        except Exception:
            continue

    freq = selection_count / n_bootstrap

    scores = {}
    for name, f in zip(predictor_names, freq):
        scores[(name, gene_names[j])] = f

    return scores


def run_tigress(expression_df, tf_list=None, n_bootstrap=1000,
                n_jobs=4, seed=0):
    """
    Run TIGRESS on an expression matrix.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Genes as columns, samples/conditions as rows.
    tf_list : list or None
        Candidate regulators.
    n_bootstrap : int
        Number of bootstrap resamples.
    n_jobs : int
    seed : int

    Returns
    -------
    dict of {(TF, TG): stability_score}
    """
    gene_names = list(expression_df.columns)
    X = expression_df.values.astype(np.float64)

    if tf_list is not None:
        tf_set = set(tf_list) & set(gene_names)
    else:
        tf_set = set(gene_names)

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_stability_selection_one_target)(
            j, X, gene_names, n_bootstrap, 0.8, seed + j
        )
        for j in range(len(gene_names))
    )

    edge_scores = {}
    for score_dict in results:
        for (tf, tg), score in score_dict.items():
            if tf in tf_set:
                edge_scores[(tf, tg)] = score

    return edge_scores
