#!/usr/bin/env python
"""
Single DREAM4 full-data benchmark task for SLURM arrays.

Each task runs all four full-data methods for one (network, seed) pair and
writes an independent TSV. The task outputs are merged by
combine_dream4_full_results.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DREAM4_DIR, DREAM4_NETS, RESULTS_DIR, SEED
from evaluation.metrics import evaluate_predictions, ranked_edge_list_to_scores
from methods import run_genie3, run_grnboost2, run_ilsgrn, run_tigress

SEEDS = [SEED + i for i in range(10)]
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

METHOD_FACTORIES = {
    "GENIE3": lambda df, s: run_genie3(df, n_trees=1000, n_jobs=N_JOBS, seed=s),
    "GRNBoost2": lambda df, s: run_grnboost2(df, n_jobs=N_JOBS, seed=s),
    "iLSGRN": lambda df, s: run_ilsgrn(df, n_jobs=N_JOBS, seed=s),
    "TIGRESS": lambda df, s: run_tigress(df, n_bootstrap=500, n_jobs=N_JOBS, seed=s),
}


def task_to_network_seed(task_id):
    n_tasks = len(DREAM4_NETS) * len(SEEDS)
    if task_id < 0 or task_id >= n_tasks:
        raise ValueError(f"task_id must be in [0, {n_tasks - 1}], got {task_id}")
    net_idx = task_id // len(SEEDS)
    seed_idx = task_id % len(SEEDS)
    return DREAM4_NETS[net_idx], SEEDS[seed_idx]


def load_gold_standard(net_id):
    gs_path = DREAM4_DIR / f"net{net_id}_goldstandard.tsv"
    gs = pd.read_csv(gs_path, sep="\t", header=None, names=["TF", "TG", "label"])
    return set(zip(gs[gs.label == 1].TF, gs[gs.label == 1].TG))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--network", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / "dream4_full_tasks",
    )
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if args.network is not None and args.seed is not None:
        net_id, seed = args.network, args.seed
    elif task_id is not None:
        net_id, seed = task_to_network_seed(task_id)
    else:
        raise SystemExit("Provide --task-id or both --network and --seed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"net{net_id}_seed{seed}.tsv"

    print("=" * 70, flush=True)
    print(f"DREAM4 full-data task: network={net_id}, seed={seed}, n_jobs={N_JOBS}", flush=True)
    print(f"Output: {out_file}", flush=True)
    print("=" * 70, flush=True)

    expr_df = pd.read_csv(DREAM4_DIR / f"net{net_id}_expression.tsv", sep="\t", index_col=0)
    gold_set = load_gold_standard(net_id)
    genes = list(expr_df.columns)

    rows = []
    for method_name, runner in METHOD_FACTORIES.items():
        print(f"Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            edge_scores = runner(expr_df, seed)
            elapsed = time.time() - t0
            y_true, y_score = ranked_edge_list_to_scores(edge_scores, gold_set, genes)
            metrics = evaluate_predictions(y_true, y_score)
            print(
                f"{method_name}: AUROC={metrics['AUROC']:.4f} "
                f"AUPRC={metrics['AUPRC']:.4f} EP@5%={metrics['EP@5%']:.4f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            rows.append({
                "network": net_id,
                "seed": seed,
                "method": method_name,
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
                "EP_5pct": metrics["EP@5%"],
                "EP_10pct": metrics["EP@10%"],
                "time_sec": elapsed,
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR: {method_name} net{net_id} seed{seed}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            rows.append({
                "network": net_id,
                "seed": seed,
                "method": method_name,
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "EP_5pct": np.nan,
                "EP_10pct": np.nan,
                "time_sec": elapsed,
            })

    tmp_file = out_file.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp_file, sep="\t", index=False)
    tmp_file.replace(out_file)
    print(f"Saved {len(rows)} rows to {out_file}", flush=True)


if __name__ == "__main__":
    main()
