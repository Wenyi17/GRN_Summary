"""
iLSGRN (Wu et al., 2023)

Wrapper around the original iLSGRN code (github.com/lab319/iLSGRN).
Uses MIC (Maximal Information Coefficient) for regulatory gene recognition,
then RF × XGBoost feature importances for edge scoring.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from minepy import MINE
from joblib import Parallel, delayed


def _compute_mic_matrix(X, gene_names):
    """Compute MIC (Maximal Information Coefficient) between all gene pairs."""
    n = len(gene_names)
    mic_mat = pd.DataFrame(np.zeros((n, n)), index=gene_names, columns=gene_names)
    mine = MINE(alpha=0.6, c=15)

    for i in range(n):
        for j in range(i + 1, n):
            mine.compute_score(X[:, i], X[:, j])
            mic_val = mine.mic()
            mic_mat.iloc[i, j] = mic_val
            mic_mat.iloc[j, i] = mic_val
        mic_mat.iloc[i, i] = 1.0

    return mic_mat


def _get_importances_single_ss(SS_data, input_idx, output_idx, alpha,
                                param_xgb, ngenes):
    """
    Original iLSGRN importance computation for steady-state data.
    Uses RF * XGBoost feature importances (multiplicative fusion).
    """
    input_matrix = SS_data[:, input_idx]
    output_vect = SS_data[:, output_idx] * alpha

    input_matrix[np.isinf(input_matrix)] = 0
    output_vect[np.isinf(output_vect)] = 0

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", max_depth=4,
        n_jobs=1, random_state=0,
    )
    rf.fit(input_matrix, output_vect)
    rf_imp = rf.feature_importances_

    # XGBoost
    xgb = XGBRegressor(**param_xgb)
    xgb.fit(input_matrix, output_vect)
    xgb_imp = xgb.feature_importances_

    fim = np.zeros(ngenes)
    fim[input_idx] = rf_imp * xgb_imp
    return fim


def _process_one_target(i, X, gene_names, mic_mat, threshold, alpha,
                         param_xgb, ngenes):
    """Process one target gene: MIC filtering + RF*XGB importance."""
    keyword1 = gene_names[i]
    idx = []
    for k in range(ngenes):
        keyword2 = gene_names[k]
        if keyword1 != keyword2 and mic_mat.loc[keyword1, keyword2] > threshold:
            idx.append(k)

    if not idx:
        return np.zeros(ngenes)

    return _get_importances_single_ss(X, idx, i, alpha, param_xgb, ngenes)


def run_ilsgrn(expression_df, tf_list=None, threshold=0.15, alpha=0.011,
               xgb_learning_rate=0.012, n_jobs=4, seed=0):
    """
    Run iLSGRN on an expression matrix (steady-state mode).

    Parameters
    ----------
    expression_df : pd.DataFrame
        Samples as rows, genes as columns.
    tf_list : list or None
    threshold : float
        MIC threshold for regulatory gene recognition.
    alpha : float
        Decay rate parameter from iLSGRN ODE model.
    xgb_learning_rate : float
    n_jobs : int
    seed : int

    Returns
    -------
    dict of {(TF, TG): importance_score}
    """
    gene_names = list(expression_df.columns)
    X = expression_df.values.astype(np.float64)
    ngenes = len(gene_names)

    if tf_list is not None:
        tf_set = set(tf_list) & set(gene_names)
    else:
        tf_set = set(gene_names)

    param_xgb = dict(
        learning_rate=xgb_learning_rate,
        importance_type="weight",
        n_estimators=500,
        max_depth=4,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )

    print("  iLSGRN: Computing MIC matrix...")
    mic_mat = _compute_mic_matrix(X, gene_names)
    print(f"  iLSGRN: MIC done. Running RF×XGB for {ngenes} targets...")

    FIM_rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_one_target)(
            i, X, gene_names, mic_mat, threshold, alpha, param_xgb, ngenes
        )
        for i in range(ngenes)
    )

    FIM = np.array(FIM_rows)

    edge_scores = {}
    for j in range(ngenes):
        if gene_names[j] not in tf_set:
            continue
        for i in range(ngenes):
            if i == j:
                continue
            edge_scores[(gene_names[j], gene_names[i])] = float(FIM[i, j])

    return edge_scores
