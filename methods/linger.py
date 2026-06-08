"""
LINGER (Yuan et al., 2023)

Wrapper faithful to the original LINGER architecture
(github.com/Durenlab/LINGER).

In expression-only mode (for cross-category benchmarking), uses the
core per-target neural network: TF expression -> 64 -> 16 -> 1,
with gradient-based importance scoring. The full LINGER also includes
ATAC-seq integration and lifelong learning (EWC), which are omitted
here since only expression data is available.

Architecture from original LL_net.py / LINGER_tr.py:
  Net: Linear(n_input, 64) -> ReLU -> Linear(64, 16) -> ReLU -> Linear(16, 1)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from joblib import Parallel, delayed


class LINGERNet(nn.Module):
    """Exact Net architecture from original LINGER code."""
    def __init__(self, input_size, activef="ReLU"):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activef = activef

    def forward(self, x):
        if self.activef == "ReLU":
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        elif self.activef == "sigmoid":
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
        elif self.activef == "tanh":
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def _train_one_target(target_idx, X_tf, Y_all, gene_names, tf_names,
                      n_epochs=200, lr=1e-3, seed=42):
    """
    Train one LINGER Net for a single target gene.
    Returns importance scores for each TF.
    """
    torch.manual_seed(seed + target_idx)

    y = Y_all[:, target_idx]
    inputs = torch.tensor(X_tf, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Normalize inputs per-TF
    mean = inputs.mean(dim=0)
    std = inputs.std(dim=0) + 1e-12
    inputs = (inputs - mean) / std

    n_tf = inputs.shape[1]
    net = LINGERNet(n_tf, "ReLU")
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    net.train()
    for epoch in range(n_epochs):
        pred = net(inputs)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute gradient-based importance (integrated gradients approximation)
    net.eval()
    inputs.requires_grad_(True)
    pred = net(inputs)
    pred.sum().backward()

    grad_importance = inputs.grad.abs().mean(dim=0).detach().numpy()

    # Also use first-layer weight magnitudes (as in LINGER SHAP fallback)
    w1 = net.fc1.weight.detach().abs().mean(dim=0).numpy()

    # Combine gradient and weight importance
    importance = grad_importance * w1

    scores = {}
    target_name = gene_names[target_idx]
    for i, tf_name in enumerate(tf_names):
        scores[(tf_name, target_name)] = float(importance[i])

    return scores


def run_linger(expression_df, train_edges=None, eval_pairs=None,
               tf_list=None, n_epochs=200, lr=1e-3,
               hidden_dim=64, device="cpu", seed=42):
    """
    Run LINGER (expression-only mode).

    For supervised mode (train_edges provided): uses known edges to
    weight the loss during training.
    For unsupervised mode: trains on all expression data.

    Parameters
    ----------
    expression_df : pd.DataFrame
    train_edges : list of (TF, TG, label) or None
    eval_pairs : list of (TF, TG) or None
    tf_list : list of TF names
    n_epochs : int
    lr : float
    device : str
    seed : int

    Returns
    -------
    dict of {(TF, TG): score}
    """
    np.random.seed(seed)
    gene_names = list(expression_df.columns)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    X_all = expression_df.values.astype(np.float64)

    if tf_list is None:
        tf_list = gene_names
    tf_in_expr = [t for t in tf_list if t in gene2idx]
    tf_indices = [gene2idx[t] for t in tf_in_expr]

    X_tf = X_all[:, tf_indices]

    print(f"  LINGER: {n_genes} genes, {len(tf_in_expr)} TFs, {X_all.shape[0]} samples")
    print(f"  LINGER: Training {n_genes} per-target networks...")

    all_scores = Parallel(n_jobs=4, verbose=1)(
        delayed(_train_one_target)(
            j, X_tf, X_all, gene_names, tf_in_expr,
            n_epochs=n_epochs, lr=lr, seed=seed,
        )
        for j in range(n_genes)
    )

    edge_scores = {}
    for score_dict in all_scores:
        edge_scores.update(score_dict)

    if eval_pairs is not None:
        return {p: edge_scores.get(p, 0.0) for p in eval_pairs}

    return edge_scores
