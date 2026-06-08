"""
GNNLink (Mao et al., 2023)

PyTorch reimplementation faithful to the original TensorFlow code
(github.com/sdesignates/GNNLink).

Architecture: 2-layer GCN encoder (input_dim -> 128 -> 64)
              + inner-product decoder (score = ReLU(Z @ Z^T))
Training: binary cross-entropy on masked positive/negative edges.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class GCNEncoder(nn.Module):
    """
    Two-layer GCN following original GNNLink:
      Layer 1: input_dim -> 128 (LeakyReLU)
      Layer 2: 128 -> 64 (LeakyReLU)
    """
    def __init__(self, input_dim, hidden_dim=128, emb_dim=64, dropout=0.25):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(hidden_dim, emb_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.dropout = dropout

    def forward(self, X, adj_sp):
        """
        X: (n_genes, n_features)  node feature matrix
        adj_sp: (n_genes, n_genes) normalized adjacency (dense or sparse)
        """
        h = F.leaky_relu(adj_sp @ (X @ self.W1))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(adj_sp @ (h @ self.W2))
        return h

    def decode(self, Z):
        """Inner product decoder: score_ij = ReLU(z_i @ z_j)."""
        logits = Z @ Z.T
        return F.relu(logits)


def _build_adj_from_edges(n_genes, edges, gene2idx):
    """Build sparse adjacency from training edges."""
    row, col = [], []
    for tf, tg, label in edges:
        if label == 1 and tf in gene2idx and tg in gene2idx:
            row.append(gene2idx[tf])
            col.append(gene2idx[tg])
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((vals, (row, col)), shape=(n_genes, n_genes)).toarray()
    return A


def _normalize_adj(A):
    """Symmetric normalization: D^{-1/2}(A+I)D^{-1/2}."""
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


def _masked_loss(logits_flat, labels, mask, neg_mask):
    """
    Reproduce GNNLink's loss: weighted BCE on masked entries.
    logits_flat: (n_genes^2,)
    labels: (n_genes^2,) binary
    mask: (n_genes^2,) 1 for positive training edges
    neg_mask: (n_genes^2,) 1 for negative training edges
    """
    pos_loss = -torch.log(logits_flat + 1e-8) * mask
    neg_loss = -torch.log(1 - logits_flat + 1e-8) * neg_mask
    n_pos = mask.sum() + 1e-8
    n_neg = neg_mask.sum() + 1e-8
    return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg


def run_gnnlink(expression_df, train_edges, eval_pairs,
                tf_list=None, n_epochs=100, lr=0.001,
                hidden_dim=128, emb_dim=64,
                device="cpu", seed=0):
    """
    Train GNNLink and score evaluation pairs.

    Parameters
    ----------
    expression_df : pd.DataFrame
    train_edges : list of (TF, TG, label)
    eval_pairs : list of (TF, TG)
    device : str
    seed : int

    Returns
    -------
    dict of {(TF, TG): score}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gene_names = list(expression_df.columns)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    # Node features = expression profiles (transposed: genes x cells)
    X_np = expression_df.values.astype(np.float32).T
    X_np = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-10)
    feat_dim = X_np.shape[1]

    A = _build_adj_from_edges(n_genes, train_edges, gene2idx)
    A_hat = _normalize_adj(A)

    # Build training masks
    train_label_mat = np.zeros(n_genes * n_genes, dtype=np.float32)
    train_mask = np.zeros(n_genes * n_genes, dtype=np.float32)
    neg_mask = np.zeros(n_genes * n_genes, dtype=np.float32)

    for tf, tg, label in train_edges:
        if tf in gene2idx and tg in gene2idx:
            flat_idx = gene2idx[tf] * n_genes + gene2idx[tg]
            if label == 1:
                train_label_mat[flat_idx] = 1.0
                train_mask[flat_idx] = 1.0
            else:
                neg_mask[flat_idx] = 1.0

    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    A_hat_t = torch.tensor(A_hat, dtype=torch.float32).to(device)
    mask_t = torch.tensor(train_mask, dtype=torch.float32).to(device)
    neg_mask_t = torch.tensor(neg_mask, dtype=torch.float32).to(device)

    model = GCNEncoder(feat_dim, hidden_dim, emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for epoch in range(n_epochs):
        Z = model(X_t, A_hat_t)
        score_mat = model.decode(Z)
        logits_flat = score_mat.reshape(-1)
        logits_flat = torch.clamp(logits_flat, 0, 1)

        loss = _masked_loss(logits_flat, None, mask_t, neg_mask_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  GNNLink epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

    # Score evaluation pairs
    model.eval()
    edge_scores = {}
    with torch.no_grad():
        Z = model(X_t, A_hat_t)
        score_mat = model.decode(Z).cpu().numpy()

        for tf, tg in eval_pairs:
            if tf in gene2idx and tg in gene2idx:
                edge_scores[(tf, tg)] = float(score_mat[gene2idx[tf], gene2idx[tg]])

    return edge_scores
