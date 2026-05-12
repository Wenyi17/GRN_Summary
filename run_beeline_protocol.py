#!/usr/bin/env python
"""
BEELINE hESC benchmark —  common evaluation protocol.

Same dataset construction and evaluation as Common/run_extra_baselines.py:
  - Gene universe: expr ∩ gold genes (capped at 5000)
  - Balanced 1:1 pos/neg pairs
  - 10-fold StratifiedKFold on pairs
  - sklearn roc_auc_score / average_precision_score

"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 42
N_FOLDS = 10
MAX_GENES = 5000
MAX_POS_PAIRS = 30000

BEELINE_EXPR = "/nas/longleaf/home/leyudai/Beeline/BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv"
BEELINE_GOLD = "/nas/longleaf/home/leyudai/Beeline/Networks/human/hESC-ChIP-seq-network.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNSUPERVISED = {"GENIE3", "GRNBoost2", "iLSGRN", "TIGRESS"}
SUPERVISED = {"DeepSEM", "DeepTGI", "GNNLink", "LINGER"}


def load_dataset():
    """Identical to Common/run_extra_baselines.py load_dataset()."""
    print("  Loading data...", flush=True)
    expr_df = pd.read_csv(BEELINE_EXPR, index_col=0)
    gold = pd.read_csv(BEELINE_GOLD)
    print(f"  Raw: {expr_df.shape[0]} genes × {expr_df.shape[1]} cells", flush=True)

    gold_genes = set(gold["Gene1"].unique()) | set(gold["Gene2"].unique())
    gold_in_expr = gold_genes & set(expr_df.index)

    if expr_df.shape[0] > MAX_GENES:
        non_gold = expr_df.index.difference(list(gold_in_expr))
        var = expr_df.loc[non_gold].var(axis=1)
        n_hvg = MAX_GENES - len(gold_in_expr)
        hvg = var.nlargest(max(0, n_hvg)).index
        keep = sorted(gold_in_expr | set(hvg))[:MAX_GENES]
        expr_df = expr_df.loc[expr_df.index.isin(keep)]

    genes = list(expr_df.index)
    g2i = {g: i for i, g in enumerate(genes)}
    n_genes, n_cells = len(genes), expr_df.shape[1]

    tf_set = set(gold["Gene1"].unique()) & set(genes)
    tf_names = sorted(tf_set)
    tf_idx = np.array([g2i[t] for t in tf_names])
    print(f"  {n_genes} genes, {n_cells} cells, {len(tf_names)} TFs", flush=True)

    expr_mat = expr_df.values.T.astype(np.float32)  # (cells, genes)

    pos_pairs = list({(g2i[r["Gene1"]], g2i[r["Gene2"]])
                      for _, r in gold.iterrows()
                      if r["Gene1"] in g2i and r["Gene2"] in g2i and r["Gene1"] in tf_set})
    if len(pos_pairs) > MAX_POS_PAIRS:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(pos_pairs), MAX_POS_PAIRS, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx]

    pos_set = set(pos_pairs)
    rng = np.random.RandomState(SEED)
    neg_pairs = []
    while len(neg_pairs) < len(pos_pairs):
        ti = int(rng.choice(tf_idx))
        gi = rng.randint(0, n_genes)
        if (ti, gi) not in pos_set and ti != gi:
            neg_pairs.append((ti, gi))
            pos_set.add((ti, gi))

    pairs = np.array(pos_pairs + neg_pairs)
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    print(f"  {len(pos_pairs)} pos + {len(neg_pairs)} neg = {len(labels)} pairs", flush=True)

    return dict(genes=genes, g2i=g2i, n_genes=n_genes, n_cells=n_cells,
                tf_names=tf_names, tf_idx=tf_idx, expr=expr_mat,
                expr_df=expr_df, pairs=pairs, labels=labels)


# ────────────────────────────────────────────────────────────
# Unsupervised method runners → score_mat (n_tfs × n_genes)
# ────────────────────────────────────────────────────────────
def run_genie3_mat(data, n_jobs=8):
    from sklearn.ensemble import RandomForestRegressor
    expr = data["expr"]  # (cells, genes)
    tf_idx = data["tf_idx"]
    ng = data["n_genes"]
    print(f"  GENIE3: {len(tf_idx)} TFs × {ng} genes", flush=True)

    def _one_gene(j):
        y = expr[:, j]
        predictors = tf_idx[tf_idx != j]
        X = expr[:, predictors]
        rf = RandomForestRegressor(n_estimators=500, max_depth=None, n_jobs=1, random_state=SEED)
        rf.fit(X, y)
        imp = rf.feature_importances_
        full = np.zeros(len(tf_idx))
        tf_list = tf_idx.tolist()
        for k, ti in enumerate(predictors):
            full[tf_list.index(ti)] = imp[k]
        return full

    scores = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one_gene)(j) for j in range(ng))
    return np.column_stack(scores)  # (n_tfs, n_genes)


def run_grnboost2_mat(data, n_jobs=8):
    from methods.grnboost2 import run_grnboost2
    expr_df = data["expr_df"]
    tf_names = data["tf_names"]
    tf_idx = data["tf_idx"]
    genes = data["genes"]
    ng = data["n_genes"]
    print(f"  GRNBoost2: {len(tf_names)} TFs × {ng} genes", flush=True)

    expr_T = expr_df.T.copy()
    expr_T.columns = genes
    edge_scores = run_grnboost2(expr_T, tf_list=tf_names, n_jobs=n_jobs, seed=SEED)

    tf_pos = {t: i for i, t in enumerate(tf_names)}
    g2i = data["g2i"]
    mat = np.zeros((len(tf_names), ng), dtype=np.float32)
    for (tf, tg), imp in edge_scores.items():
        if tf in tf_pos and tg in g2i:
            mat[tf_pos[tf], g2i[tg]] = imp
    return mat


def run_tigress_mat(data, n_jobs=8):
    from sklearn.linear_model import lars_path
    expr = data["expr"]
    tf_idx = data["tf_idx"]
    ng = data["n_genes"]
    n_bootstrap = 200
    print(f"  TIGRESS: {len(tf_idx)} TFs × {ng} genes, {n_bootstrap} bootstraps", flush=True)

    def _one_gene(j):
        y = expr[:, j]
        predictors = tf_idx[tf_idx != j]
        X = expr[:, predictors]
        n_samples = X.shape[0]
        freq = np.zeros(len(predictors))
        rng = np.random.RandomState(SEED + j)
        for b in range(n_bootstrap):
            idx = rng.choice(n_samples, n_samples, replace=True)
            try:
                _, _, coefs = lars_path(X[idx], y[idx], method='lasso', max_iter=min(3, len(predictors)))
                selected = np.where(np.abs(coefs[:, -1]) > 0)[0]
                freq[selected] += 1
            except Exception:
                pass
        freq /= n_bootstrap
        full = np.zeros(len(tf_idx))
        tf_list = tf_idx.tolist()
        for k, ti in enumerate(predictors):
            full[tf_list.index(ti)] = freq[k]
        return full

    scores = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one_gene)(j) for j in range(ng))
    return np.column_stack(scores)


def run_ilsgrn_mat(data, n_jobs=8):
    """iLSGRN: XGBoost × RF importance (without MIC filter for speed)."""
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    expr = data["expr"]
    tf_idx = data["tf_idx"]
    ng = data["n_genes"]
    print(f"  iLSGRN: {len(tf_idx)} TFs × {ng} genes", flush=True)

    def _one_gene(j):
        y = expr[:, j]
        predictors = tf_idx[tf_idx != j]
        X = expr[:, predictors]
        rf = RandomForestRegressor(n_estimators=500, max_depth=4, n_jobs=1, random_state=SEED)
        rf.fit(X, y)
        xm = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.012,
                               objective="reg:squarederror", n_jobs=1, verbosity=0, random_state=SEED)
        xm.fit(X, y)
        fused = rf.feature_importances_ * xm.feature_importances_
        full = np.zeros(len(tf_idx))
        tf_list = tf_idx.tolist()
        for k, ti in enumerate(predictors):
            full[tf_list.index(ti)] = fused[k]
        return full

    scores = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one_gene)(j) for j in range(ng))
    return np.column_stack(scores)


UNSUP_RUNNERS = {
    "GENIE3": run_genie3_mat,
    "GRNBoost2": run_grnboost2_mat,
    "TIGRESS": run_tigress_mat,
    "iLSGRN": run_ilsgrn_mat,
}


# ────────────────────────────────────────────────────────────
# Supervised method runners → per-fold scores
# ────────────────────────────────────────────────────────────
def run_supervised_fold(method_name, data, train_pairs, train_labels, test_pairs):
    """Train on fold's train pairs, return scores for test pairs."""
    expr = data["expr"]  # (cells, genes)
    genes = data["genes"]
    tf_names = data["tf_names"]
    g2i = data["g2i"]
    ng = data["n_genes"]
    idx2gene = {i: g for g, i in g2i.items()}

    expr_df_T = pd.DataFrame(expr, columns=genes)

    tf_list_method = tf_names
    train_edges = [(idx2gene[int(p[0])], idx2gene[int(p[1])], int(l))
                   for p, l in zip(train_pairs, train_labels)]

    fold_seed = SEED + hash(str(train_pairs[:3])) % 10000

    if method_name == "DeepSEM":
        from methods import run_deepsem
        edge_scores = run_deepsem(expr_df_T, tf_list=tf_list_method, n_epochs=100,
                                  lr=1e-4, hidden_dim=128, device=DEVICE, seed=fold_seed)
        scores = np.array([edge_scores.get((idx2gene[int(p[0])], idx2gene[int(p[1])]), 0.0)
                           for p in test_pairs])
    elif method_name == "DeepTGI":
        from methods import run_deeptgi
        gold_train = set((tf, tg) for tf, tg, l in train_edges if l == 1)
        train_tfs = list(set(tf for tf, _, l in train_edges if l == 1))
        eval_tfs = list(set(idx2gene[int(p[0])] for p in test_pairs))
        edge_scores = run_deeptgi(expr_df_T, gold_edges=gold_train, tf_list=tf_list_method,
                                  train_tfs=train_tfs, eval_tfs=eval_tfs,
                                  n_epochs=40, lr=1e-5, batch_size=256,
                                  proj_dim=512, device=DEVICE, seed=fold_seed)
        scores = np.array([edge_scores.get((idx2gene[int(p[0])], idx2gene[int(p[1])]), 0.0)
                           for p in test_pairs])
    elif method_name == "GNNLink":
        from methods import run_gnnlink
        eval_pairs_named = [(idx2gene[int(p[0])], idx2gene[int(p[1])]) for p in test_pairs]
        edge_scores = run_gnnlink(expr_df_T, train_edges=train_edges, eval_pairs=eval_pairs_named,
                                  tf_list=tf_list_method, n_epochs=100, lr=1e-4,
                                  hidden_dim=128, emb_dim=64, device=DEVICE, seed=fold_seed)
        scores = np.array([edge_scores.get(p, 0.0) for p in eval_pairs_named])
    elif method_name == "LINGER":
        from methods import run_linger
        eval_pairs_named = [(idx2gene[int(p[0])], idx2gene[int(p[1])]) for p in test_pairs]
        edge_scores = run_linger(expr_df_T, train_edges=train_edges, eval_pairs=eval_pairs_named,
                                 tf_list=tf_list_method, n_epochs=100, lr=1e-4,
                                 hidden_dim=128, device=DEVICE, seed=fold_seed)
        scores = np.array([edge_scores.get(p, 0.0) for p in eval_pairs_named])
    else:
        raise ValueError(f"Unknown supervised method: {method_name}")

    return scores


# ────────────────────────────────────────────────────────────
# Evaluation (identical to Common protocol)
# ────────────────────────────────────────────────────────────
def eval_unsupervised(method_name, score_mat, data):
    """Evaluate unsupervised method using Common's protocol."""
    pairs, labels = data["pairs"], data["labels"]
    tf_idx = data["tf_idx"]
    tf_pos = {int(v): i for i, v in enumerate(tf_idx)}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (_, te_idx) in enumerate(skf.split(pairs, labels)):
        tp, tl = pairs[te_idx], labels[te_idx]
        sc = np.array([score_mat[tf_pos.get(int(p[0]), 0), int(p[1])] for p in tp])
        auc = roc_auc_score(tl, sc)
        apr = average_precision_score(tl, sc)
        print(f"    Fold {fold+1:2d}: AUROC={auc:.4f}  AUPRC={apr:.4f}", flush=True)
        rows.append({"method": method_name, "fold": fold + 1, "auroc": auc, "auprc": apr})
    return rows


def eval_supervised(method_name, data):
    """Evaluate supervised method per fold using Common's protocol."""
    pairs, labels = data["pairs"], data["labels"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(pairs, labels)):
        print(f"    Fold {fold+1:2d}: training...", end=" ", flush=True)
        t0 = time.time()
        try:
            sc = run_supervised_fold(method_name, data,
                                     pairs[tr_idx], labels[tr_idx],
                                     pairs[te_idx])
            auc = roc_auc_score(labels[te_idx], sc)
            apr = average_precision_score(labels[te_idx], sc)
            print(f"AUROC={auc:.4f}  AUPRC={apr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"method": method_name, "fold": fold + 1, "auroc": auc, "auprc": apr})
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            rows.append({"method": method_name, "fold": fold + 1, "auroc": np.nan, "auprc": np.nan})
    return rows


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60, flush=True)
    print(f"  BEELINE hESC — Common protocol", flush=True)
    print(f"  Methods: {methods}", flush=True)
    print(f"  {N_FOLDS}-fold StratifiedKFold, balanced 1:1 pairs", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    data = load_dataset()
    all_rows = []

    for method_name in methods:
        print(f"\n{'━' * 60}", flush=True)
        print(f"  {method_name}", flush=True)
        print(f"{'━' * 60}", flush=True)
        t0 = time.time()

        if method_name in UNSUPERVISED:
            runner = UNSUP_RUNNERS[method_name]
            score_mat = runner(data)
            rows = eval_unsupervised(method_name, score_mat, data)
        elif method_name in SUPERVISED:
            rows = eval_supervised(method_name, data)
        else:
            print(f"  Unknown method: {method_name}", flush=True)
            continue

        all_rows.extend(rows)
        elapsed = time.time() - t0

        aucs = [r["auroc"] for r in rows if not np.isnan(r["auroc"])]
        aprs = [r["auprc"] for r in rows if not np.isnan(r["auprc"])]
        print(f"\n  {method_name}: AUROC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
              f"AUPRC={np.mean(aprs):.4f}±{np.std(aprs):.4f}  ({elapsed/3600:.1f}h)", flush=True)

        pd.DataFrame(all_rows).to_csv(out_path, sep="\t", index=False)
        print(f"  Saved {len(all_rows)} rows to {out_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print("  " + "─" * 56, flush=True)
    df = pd.DataFrame(all_rows)
    for m in methods:
        sub = df[df.method == m]
        if len(sub) == 0: continue
        print(f"  {m:<12}  AUROC={sub.auroc.mean():.4f}±{sub.auroc.std():.4f}  "
              f"AUPRC={sub.auprc.mean():.4f}±{sub.auprc.std():.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
