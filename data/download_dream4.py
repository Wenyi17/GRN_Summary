#!/usr/bin/env python
"""
Prepare DREAM4 data from the user-downloaded DREAM4data/ directory.

Source: /nas/longleaf/home/leyudai/DREAM4data/
  - Multifactorial expression: DREAM4_InSilico_Size100_Multifactorial (1)/
  - Gold standard: DREAM4_InSilicoNetworks_GoldStandard/.../Size 100 multifactorial/

Creates:
  data/DREAM4/
    net{1..5}_expression.tsv   (100 conditions x 100 genes)
    net{1..5}_goldstandard.tsv  (TF  TG  label)
  data/DREAM4/partitioned/
    all_expression.tsv          (pooled expression, 500 x 100)
    train_edges.tsv / val_edges.tsv / test_edges.tsv  (pooled 60/20/20 split)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DREAM4_DIR, DREAM4_NETS

DREAM4_DIR.mkdir(parents=True, exist_ok=True)

SRC = Path("/nas/longleaf/home/leyudai/DREAM4data")
EXPR_DIR = SRC / "DREAM4_InSilico_Size100_Multifactorial (1)"
GS_DIR = SRC / "DREAM4_InSilicoNetworks_GoldStandard" / "DREAM4_Challenge2_GoldStandards" / "Size 100 multifactorial"


def prepare_expression():
    print("\n[1] Preparing expression data...")
    for net_id in DREAM4_NETS:
        src = EXPR_DIR / f"insilico_size100_{net_id}_multifactorial.tsv"
        dst = DREAM4_DIR / f"net{net_id}_expression.tsv"

        df = pd.read_csv(src, sep="\t")
        df.columns = [c.strip('"') for c in df.columns]
        df.index = [f"cond{i+1}" for i in range(len(df))]
        df.to_csv(dst, sep="\t")
        print(f"  Net {net_id}: {df.shape}")


def prepare_gold_standards():
    print("\n[2] Preparing gold standards...")
    for net_id in DREAM4_NETS:
        src = GS_DIR / f"DREAM4_GoldStandard_InSilico_Size100_multifactorial_{net_id}.tsv"
        dst = DREAM4_DIR / f"net{net_id}_goldstandard.tsv"

        gs = pd.read_csv(src, sep="\t", header=None, names=["TF", "TG", "label"])
        gs.to_csv(dst, sep="\t", index=False, header=False)
        n_pos = (gs.label == 1).sum()
        print(f"  Net {net_id}: {n_pos} positive / {len(gs)} total")


def prepare_partitioned_splits(seed=42):
    """Pool all 5 networks' gold standard edges and split 60/20/20."""
    print("\n[3] Creating partitioned splits (60/20/20)...")
    part_dir = DREAM4_DIR / "partitioned"
    part_dir.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_expr = []

    for net_id in DREAM4_NETS:
        gs = pd.read_csv(DREAM4_DIR / f"net{net_id}_goldstandard.tsv",
                         sep="\t", header=None, names=["TF", "TG", "label"])
        gs["network"] = net_id
        all_edges.append(gs)

        expr = pd.read_csv(DREAM4_DIR / f"net{net_id}_expression.tsv",
                           sep="\t", index_col=0)
        expr.index = [f"net{net_id}_{idx}" for idx in expr.index]
        all_expr.append(expr)

    pooled = pd.concat(all_edges, ignore_index=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pooled))

    n_train = int(len(pooled) * 0.6)
    n_val = int(len(pooled) * 0.2)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_df = pooled.iloc[train_idx][["TF", "TG", "label"]].reset_index(drop=True)
    val_df = pooled.iloc[val_idx][["TF", "TG", "label"]].reset_index(drop=True)
    test_df = pooled.iloc[test_idx][["TF", "TG", "label"]].reset_index(drop=True)

    train_df.to_csv(part_dir / "train_edges.tsv", sep="\t", index=False)
    val_df.to_csv(part_dir / "val_edges.tsv", sep="\t", index=False)
    test_df.to_csv(part_dir / "test_edges.tsv", sep="\t", index=False)

    pooled_expr = pd.concat(all_expr)
    pooled_expr.to_csv(part_dir / "all_expression.tsv", sep="\t")

    print(f"  Train: {len(train_df)} ({train_df.label.sum()} pos, {(train_df.label==0).sum()} neg)")
    print(f"  Val:   {len(val_df)} ({val_df.label.sum()} pos, {(val_df.label==0).sum()} neg)")
    print(f"  Test:  {len(test_df)} ({test_df.label.sum()} pos, {(test_df.label==0).sum()} neg)")
    print(f"  Pooled expression: {pooled_expr.shape}")


def verify():
    print("\n[4] Verifying...")
    for net_id in DREAM4_NETS:
        expr = pd.read_csv(DREAM4_DIR / f"net{net_id}_expression.tsv", sep="\t", index_col=0)
        gs = pd.read_csv(DREAM4_DIR / f"net{net_id}_goldstandard.tsv", sep="\t",
                         header=None, names=["TF", "TG", "label"])
        n_pos = (gs.label == 1).sum()
        print(f"  Net {net_id}: expr={expr.shape}, {n_pos} positive / {len(gs)} total edges")


def main():
    print("=" * 60)
    print("  DREAM4 Data Preparation (from downloaded data)")
    print("=" * 60)

    prepare_expression()
    prepare_gold_standards()
    prepare_partitioned_splits()
    verify()
    print("\nDone!")


if __name__ == "__main__":
    main()
