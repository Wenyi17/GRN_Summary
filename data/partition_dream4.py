#!/usr/bin/env python
"""
Generate DREAM4 partitioned evaluation splits.

Pools gold-standard edges from all 5 networks and splits into
non-overlapping train/val/test sets. This simulates a realistic
scenario where only partial regulatory information is available.

Creates:
  data/DREAM4/partitioned/
    train_edges.tsv
    val_edges.tsv
    test_edges.tsv
    all_expression.tsv   (concatenated, with network id column)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DREAM4_DIR, DREAM4_NETS, DREAM4_N_GENES,
                    PARTITION_SEED, PARTITION_TRAIN, PARTITION_VAL, PARTITION_TEST)

OUT_DIR = DREAM4_DIR / "partitioned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  DREAM4 Partitioned Split Generation")
    print("=" * 60)

    rng = np.random.default_rng(PARTITION_SEED)

    all_pos_edges = []
    all_neg_edges = []

    for net_id in DREAM4_NETS:
        gs_path = DREAM4_DIR / f"net{net_id}_goldstandard.tsv"
        gs = pd.read_csv(gs_path, sep="\t", header=None, names=["TF", "TG", "label"])
        gs["net_id"] = net_id

        pos = gs[gs.label == 1].copy()
        neg = gs[gs.label == 0].copy()

        all_pos_edges.append(pos)
        all_neg_edges.append(neg)
        print(f"  Net {net_id}: {len(pos)} positive, {len(neg)} negative edges")

    pos_df = pd.concat(all_pos_edges, ignore_index=True)
    neg_df = pd.concat(all_neg_edges, ignore_index=True)

    print(f"\n  Total positive edges: {len(pos_df)}")
    print(f"  Total negative edges: {len(neg_df)}")

    # Shuffle and split positive edges
    pos_idx = rng.permutation(len(pos_df))
    n_train = int(len(pos_df) * PARTITION_TRAIN)
    n_val   = int(len(pos_df) * PARTITION_VAL)

    train_pos = pos_df.iloc[pos_idx[:n_train]]
    val_pos   = pos_df.iloc[pos_idx[n_train:n_train+n_val]]
    test_pos  = pos_df.iloc[pos_idx[n_train+n_val:]]

    # For each split, sample an equal number of negatives
    neg_idx = rng.permutation(len(neg_df))
    n_neg_train = len(train_pos)
    n_neg_val   = len(val_pos)
    n_neg_test  = len(test_pos)

    cursor = 0
    train_neg = neg_df.iloc[neg_idx[cursor:cursor+n_neg_train]]; cursor += n_neg_train
    val_neg   = neg_df.iloc[neg_idx[cursor:cursor+n_neg_val]];   cursor += n_neg_val
    test_neg  = neg_df.iloc[neg_idx[cursor:cursor+n_neg_test]];  cursor += n_neg_test

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    val_df   = pd.concat([val_pos,   val_neg],   ignore_index=True)
    test_df  = pd.concat([test_pos,  test_neg],  ignore_index=True)

    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=PARTITION_SEED).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=PARTITION_SEED+1).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=PARTITION_SEED+2).reset_index(drop=True)

    train_df.to_csv(OUT_DIR / "train_edges.tsv", sep="\t", index=False)
    val_df.to_csv(OUT_DIR / "val_edges.tsv",     sep="\t", index=False)
    test_df.to_csv(OUT_DIR / "test_edges.tsv",    sep="\t", index=False)

    print(f"\n  Train: {len(train_df)} ({train_pos.shape[0]} pos + {n_neg_train} neg)")
    print(f"  Val:   {len(val_df)} ({val_pos.shape[0]} pos + {n_neg_val} neg)")
    print(f"  Test:  {len(test_df)} ({test_pos.shape[0]} pos + {n_neg_test} neg)")

    # Concatenate expression data with network indicator
    print("\n  Concatenating expression matrices...")
    expr_frames = []
    for net_id in DREAM4_NETS:
        expr = pd.read_csv(DREAM4_DIR / f"net{net_id}_expression.tsv",
                           sep="\t", index_col=0)
        expr.index = [f"net{net_id}_{idx}" for idx in expr.index]
        expr_frames.append(expr)

    all_expr = pd.concat(expr_frames, axis=0)
    all_expr.to_csv(OUT_DIR / "all_expression.tsv", sep="\t")
    print(f"  Combined expression: {all_expr.shape}")

    print("\nDone! Files saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
