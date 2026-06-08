#!/usr/bin/env python
"""Merge DREAM4 full-data SLURM array task outputs."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DREAM4_NETS, RESULTS_DIR, SEED

SEEDS = [SEED + i for i in range(10)]
METHOD_ORDER = ["GENIE3", "GRNBoost2", "iLSGRN", "TIGRESS"]
TASK_DIR = RESULTS_DIR / "dream4_full_tasks"
OUT_FILE = RESULTS_DIR / "dream4_full_results.tsv"
EXPECTED_FILES = len(DREAM4_NETS) * len(SEEDS)
EXPECTED_ROWS = EXPECTED_FILES * len(METHOD_ORDER)


def main():
    files = sorted(TASK_DIR.glob("net*_seed*.tsv"))
    if len(files) != EXPECTED_FILES:
        missing = []
        for net_id in DREAM4_NETS:
            for seed in SEEDS:
                path = TASK_DIR / f"net{net_id}_seed{seed}.tsv"
                if not path.exists():
                    missing.append(path.name)
        raise SystemExit(
            f"Expected {EXPECTED_FILES} task files, found {len(files)}. "
            f"Missing: {', '.join(missing[:20])}"
        )

    frames = [pd.read_csv(path, sep="\t") for path in files]
    df = pd.concat(frames, ignore_index=True)
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}
    df["_method_rank"] = df["method"].map(method_rank)
    df = df.sort_values(["network", "seed", "_method_rank"]).drop(columns="_method_rank")
    df.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"Saved merged results to {OUT_FILE}")
    print(f"Rows: {len(df)} ({len(DREAM4_NETS)} networks x {len(SEEDS)} seeds x {len(METHOD_ORDER)} methods)")
    summary = df.groupby("method")[["AUROC", "AUPRC", "EP_5pct"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
