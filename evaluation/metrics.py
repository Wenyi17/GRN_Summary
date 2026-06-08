"""
Evaluation metrics for GRN inference benchmarking.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def auroc(y_true, y_score):
    """Area Under the Receiver Operating Characteristic curve."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return roc_auc_score(y_true, y_score)


def auprc(y_true, y_score):
    """Area Under the Precision-Recall curve."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return np.nan
    return average_precision_score(y_true, y_score)


def early_precision(y_true, y_score, k=0.05):
    """Precision among the top k fraction of ranked predictions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_top = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return float(np.mean(y_true[top_idx]))


def evaluate_predictions(y_true, y_score):
    """
    Compute all metrics for a set of predictions.
    Returns dict with AUROC, AUPRC, EP@5%, EP@10%.
    """
    return {
        "AUROC":  auroc(y_true, y_score),
        "AUPRC":  auprc(y_true, y_score),
        "EP@5%":  early_precision(y_true, y_score, k=0.05),
        "EP@10%": early_precision(y_true, y_score, k=0.10),
    }


def ranked_edge_list_to_scores(edge_scores, gold_edges, all_genes):
    """
    Convert a dict of {(TF, TG): score} to aligned arrays (y_true, y_score)
    evaluated against gold standard edges over all possible TF-TG pairs.
    """
    gold_set = set(gold_edges)

    y_true = []
    y_score = []

    for tf in all_genes:
        for tg in all_genes:
            if tf == tg:
                continue
            y_true.append(1 if (tf, tg) in gold_set else 0)
            y_score.append(edge_scores.get((tf, tg), 0.0))

    return np.array(y_true), np.array(y_score)
