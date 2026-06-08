#!/usr/bin/env python
"""
Experiment 1: DREAM4 Full-Data Benchmark

Methods: GENIE3, GRNBoost2, iLSGRN, TIGRESS
Data: DREAM4 In Silico Size 100 Multifactorial (5 networks)
Evaluation: AUROC, AUPRC, EP@5% per (network, seed).
  5 networks × 10 seeds = 50 data points per method for boxplots.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DREAM4_DIR, DREAM4_NETS, RESULTS_DIR, SEED
from methods import run_genie3, run_grnboost2, run_ilsgrn, run_tigress
from evaluation.metrics import evaluate_predictions, ranked_edge_list_to_scores

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / "dream4_full_results.tsv"

SEEDS = [SEED + i for i in range(10)]

METHOD_FACTORIES = {
    "GENIE3":    lambda df, s: run_genie3(df, n_trees=1000, n_jobs=8, seed=s),
    "GRNBoost2": lambda df, s: run_grnboost2(df, n_jobs=8, seed=s),
    "iLSGRN":    lambda df, s: run_ilsgrn(df, n_jobs=8, seed=s),
    "TIGRESS":   lambda df, s: run_tigress(df, n_bootstrap=500, n_jobs=8, seed=s),
}


def load_gold_standard(net_id):
    gs_path = DREAM4_DIR / f"net{net_id}_goldstandard.tsv"
    gs = pd.read_csv(gs_path, sep="\t", header=None, names=["TF", "TG", "label"])
    pos = set(zip(gs[gs.label == 1].TF, gs[gs.label == 1].TG))
    return pos


def main():
    print("=" * 70)
    print("  DREAM4 Full-Data Benchmark")
    print(f"  5 networks × {len(SEEDS)} seeds = {5 * len(SEEDS)} data points per method")
    print("=" * 70)

    all_results = []

    for net_id in DREAM4_NETS:
        print(f"\n{'─' * 60}")
        print(f"  Network {net_id}")
        print(f"{'─' * 60}")

        expr_path = DREAM4_DIR / f"net{net_id}_expression.tsv"
        expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
        gold_set = load_gold_standard(net_id)
        genes = list(expr_df.columns)
        print(f"  Expression: {expr_df.shape}  |  {len(gold_set)} positive edges")

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  --- Seed {seed} (run {seed_idx+1}/{len(SEEDS)}) ---")

            for method_name, runner in METHOD_FACTORIES.items():
                print(f"  Running {method_name} (seed={seed})...")
                t0 = time.time()

                try:
                    edge_scores = runner(expr_df, seed)
                    elapsed = time.time() - t0

                    y_true, y_score = ranked_edge_list_to_scores(
                        edge_scores, gold_set, genes
                    )
                    metrics = evaluate_predictions(y_true, y_score)

                    print(f"    {elapsed:.1f}s  AUROC={metrics['AUROC']:.4f}  "
                          f"AUPRC={metrics['AUPRC']:.4f}  EP@5%={metrics['EP@5%']:.4f}")

                    all_results.append({
                        "network": net_id,
                        "seed": seed,
                        "method": method_name,
                        "AUROC": metrics["AUROC"],
                        "AUPRC": metrics["AUPRC"],
                        "EP_5pct": metrics["EP@5%"],
                        "EP_10pct": metrics["EP@10%"],
                        "time_sec": elapsed,
                    })
                except Exception as e:
                    print(f"  ERROR: {method_name} net{net_id} seed{seed}: {e}")
                    import traceback; traceback.print_exc()
                    all_results.append({
                        "network": net_id, "seed": seed,
                        "method": method_name,
                        "AUROC": np.nan, "AUPRC": np.nan,
                        "EP_5pct": np.nan, "EP_10pct": np.nan,
                        "time_sec": np.nan,
                    })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {OUT_FILE}")
    print(f"  {len(results_df)} rows  ({len(SEEDS)*len(DREAM4_NETS)} per method)")
    print(f"{'=' * 70}")

    summary = results_df.groupby("method")[["AUROC", "AUPRC", "EP_5pct"]].agg(["mean", "std"])
    print(f"\n  SUMMARY (mean ± std across {len(SEEDS)*len(DREAM4_NETS)} runs)")
    print("  " + "─" * 60)
    for method in METHOD_FACTORIES:
        row = summary.loc[method]
        print(f"  {method:<12}  AUROC={row[('AUROC','mean')]:.4f}±{row[('AUROC','std')]:.4f}  "
              f"AUPRC={row[('AUPRC','mean')]:.4f}±{row[('AUPRC','std')]:.4f}  "
              f"EP@5%={row[('EP_5pct','mean')]:.4f}±{row[('EP_5pct','std')]:.4f}")


if __name__ == "__main__":
    main()
