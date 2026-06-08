#!/usr/bin/env python
"""
Experiment 2: DREAM4 Partitioned Benchmark

Methods: LINGER, GNNLink, DeepSEM, DeepTGI
Data: Pooled DREAM4 gold-standard edges split into train/val/test.
Evaluation: AUROC, AUPRC on held-out test edges.
  10 random seeds → 10 data points per method for boxplots.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DREAM4_DIR, RESULTS_DIR, SEED, EPOCHS_DEEPSEM,
                    EPOCHS_DEEPTGI, HIDDEN_DIM, LR, BATCH_SIZE)
from methods import run_deepsem, run_deeptgi, run_gnnlink, run_linger
from evaluation.metrics import evaluate_predictions

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PART_DIR = DREAM4_DIR / "partitioned"
OUT_FILE = RESULTS_DIR / "dream4_partitioned_results.tsv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [SEED + i for i in range(10)]


def load_partitioned_data():
    train_df = pd.read_csv(PART_DIR / "train_edges.tsv", sep="\t")
    val_df   = pd.read_csv(PART_DIR / "val_edges.tsv",   sep="\t")
    test_df  = pd.read_csv(PART_DIR / "test_edges.tsv",  sep="\t")
    expr_df  = pd.read_csv(PART_DIR / "all_expression.tsv", sep="\t", index_col=0)
    return train_df, val_df, test_df, expr_df


def main():
    print("=" * 70)
    print("  DREAM4 Partitioned Benchmark")
    print(f"  Device: {DEVICE}  |  {len(SEEDS)} seeds → 10 data points per method")
    print("=" * 70)

    train_df, val_df, test_df, expr_df = load_partitioned_data()

    print(f"  Expression: {expr_df.shape}")
    print(f"  Train: {len(train_df)} ({train_df.label.sum()} pos)")
    print(f"  Val:   {len(val_df)} ({val_df.label.sum()} pos)")
    print(f"  Test:  {len(test_df)} ({test_df.label.sum()} pos)")

    train_val = pd.concat([train_df, val_df])
    train_val_tuples = list(zip(train_val.TF, train_val.TG, train_val.label))
    test_pairs = list(zip(test_df.TF, test_df.TG))
    test_labels = test_df.label.values

    gold_set_all = set(
        (r.TF, r.TG) for _, r in train_val[train_val.label == 1].iterrows()
    )

    all_results = []

    methods = {
        "DeepSEM": lambda seed: run_deepsem(
            expr_df, tf_list=None, n_epochs=EPOCHS_DEEPSEM,
            lr=LR, hidden_dim=HIDDEN_DIM, device=DEVICE, seed=seed,
        ),
        "DeepTGI": lambda seed: run_deeptgi(
            expr_df, gold_edges=gold_set_all, tf_list=None,
            n_epochs=EPOCHS_DEEPTGI, lr=1e-5, batch_size=BATCH_SIZE,
            proj_dim=512, device=DEVICE, seed=seed,
        ),
        "GNNLink": lambda seed: run_gnnlink(
            expr_df, train_edges=train_val_tuples, eval_pairs=test_pairs,
            n_epochs=200, lr=LR, hidden_dim=HIDDEN_DIM, emb_dim=64,
            device=DEVICE, seed=seed,
        ),
        "LINGER": lambda seed: run_linger(
            expr_df, train_edges=train_val_tuples, eval_pairs=test_pairs,
            n_epochs=200, lr=LR, hidden_dim=HIDDEN_DIM,
            device=DEVICE, seed=seed,
        ),
    }

    for method_name, runner in methods.items():
        print(f"\n{'─' * 60}")
        print(f"  {method_name}  ({len(SEEDS)} runs)")
        print(f"{'─' * 60}")

        for run_idx, seed in enumerate(SEEDS):
            print(f"\n  Run {run_idx+1}/{len(SEEDS)} (seed={seed})...")
            t0 = time.time()
            try:
                edge_scores = runner(seed)
                elapsed = time.time() - t0
                test_scores = np.array([edge_scores.get(p, 0.0) for p in test_pairs])
                metrics = evaluate_predictions(test_labels, test_scores)
                print(f"    {elapsed:.1f}s  AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}")
                all_results.append({
                    "method": method_name, "seed": seed, "run": run_idx + 1,
                    **metrics, "time_sec": elapsed,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    ERROR ({elapsed:.1f}s): {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    "method": method_name, "seed": seed, "run": run_idx + 1,
                    "AUROC": np.nan, "AUPRC": np.nan,
                    "EP@5%": np.nan, "EP@10%": np.nan,
                    "time_sec": elapsed,
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {OUT_FILE}")
    print(f"  {len(results_df)} rows  ({len(SEEDS)} per method)")
    print(f"{'=' * 70}")

    summary = results_df.groupby("method")[["AUROC", "AUPRC"]].agg(["mean", "std"])
    print(f"\n  SUMMARY (mean ± std across {len(SEEDS)} runs)")
    print("  " + "─" * 60)
    for method in methods:
        row = summary.loc[method]
        print(f"  {method:<12}  AUROC={row[('AUROC','mean')]:.4f}±{row[('AUROC','std')]:.4f}  "
              f"AUPRC={row[('AUPRC','mean')]:.4f}±{row[('AUPRC','std')]:.4f}")


if __name__ == "__main__":
    main()
