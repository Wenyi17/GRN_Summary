"""
GENIE3 (Huynh-Thu et al., 2010)

Faithful reimplementation of the GENIE3 algorithm:
For each target gene j, fit a Random Forest to predict j from all
other genes (or candidate TFs), then use feature importances as
regulatory edge weights.

This is algorithmically identical to the arboreto/R GENIE3 package.
The arboreto package wraps the same sklearn RandomForestRegressor;
we call sklearn directly to avoid dask compatibility issues on HPC.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


def _fit_one_target(j, X, gene_names, tf_mask, n_trees, seed):
    """Fit RF for a single target gene, return importance vector."""
    y = X[:, j]
    mask = np.ones(X.shape[1], dtype=bool)
    mask[j] = False
    X_pred = X[:, mask]
    predictor_names = [g for k, g in enumerate(gene_names) if k != j]

    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X_pred, y)

    importances = {}
    for name, imp in zip(predictor_names, rf.feature_importances_):
        if tf_mask is None or name in tf_mask:
            importances[(name, gene_names[j])] = imp

    return importances


def run_genie3(expression_df, tf_list=None, n_trees=1000, n_jobs=4, seed=0):
    """
    Run GENIE3.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Genes as columns, samples/conditions as rows.
    tf_list : list or None
        Candidate regulators. None = all genes.
    n_trees : int
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
        delayed(_fit_one_target)(j, X, gene_names, tf_mask, n_trees, seed + j)
        for j in range(len(gene_names))
    )

    edge_scores = {}
    for imp_dict in results:
        edge_scores.update(imp_dict)

    return edge_scores
