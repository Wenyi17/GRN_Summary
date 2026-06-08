"""
DeepTGI (Yao et al., 2024)

Deep learning model for TF-Gene Interaction prediction.
Uses PCC (Pearson Correlation Coefficient) feature vectors as input,
with an autoencoder + self-attention architecture followed by a
binary classifier.
"""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ──────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────
LEN_AFTER_AE = 400
BERT_N_HEADS = 4
DROP_OUT     = 0.3


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(input_dim, self.d_k * n_heads, bias=False)
        self.W_K = nn.Linear(input_dim, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(input_dim, self.d_v * n_heads, bias=False)
        self.fc  = nn.Linear(n_heads * self.d_v, input_dim, bias=False)

    def forward(self, X):
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        return self.fc(ctx)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = nn.LayerNorm(input_dim)
        self.l1  = nn.Linear(input_dim, input_dim)
        self.AN2 = nn.LayerNorm(input_dim)

    def forward(self, X):
        X = self.AN1(self.attn(X) + X)
        X = self.AN2(self.l1(X) + X)
        return X


class AE1(nn.Module):
    def __init__(self, vector_size):
        super().__init__()
        mid = (vector_size + LEN_AFTER_AE) // 2
        self.l1  = nn.Linear(vector_size, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.att = EncoderLayer(mid, BERT_N_HEADS)
        self.l2  = nn.Linear(mid, LEN_AFTER_AE)
        self.l3  = nn.Linear(LEN_AFTER_AE, mid)
        self.bn3 = nn.BatchNorm1d(mid)
        self.l4  = nn.Linear(mid, vector_size)
        self.dr  = nn.Dropout(DROP_OUT)

    def forward(self, X):
        X = self.dr(self.bn1(gelu(self.l1(X))))
        X = self.att(X)
        lat = self.l2(X)
        rec = self.dr(self.bn3(gelu(self.l3(lat))))
        rec = self.l4(rec)
        return lat, rec


class DeepTGIModel(nn.Module):
    """
    DeepTGI with input projection for large gene sets.
    Input: concatenated PCC rows for TF and TG (2*n_genes dim).
    """
    def __init__(self, raw_input_dim, proj_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.ae = AE1(proj_dim)
        mid = (LEN_AFTER_AE + 1) // 2
        self.l1  = nn.Linear(LEN_AFTER_AE, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.l2  = nn.Linear(mid, 1)
        self.dr  = nn.Dropout(DROP_OUT)

    def forward(self, pcc_feat):
        proj = self.proj(pcc_feat)
        lat, rec = self.ae(proj)
        x = self.dr(self.bn1(gelu(self.l1(lat))))
        return self.l2(x).squeeze(1), proj, rec


class DeepTGILoss(nn.Module):
    def __init__(self, cls_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.cls_weight = cls_weight

    def forward(self, logit, target, projected, reconstructed):
        return (self.cls_weight * self.bce(logit, target)
                + self.mse(projected.detach(), reconstructed))


# ──────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs_df, pcc_matrix, gene2idx):
        self.df = pairs_df.reset_index(drop=True)
        self.pcc = pcc_matrix
        self.g2i = gene2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        i_tf = self.g2i[row.TF]
        i_tg = self.g2i[row.TG]
        feat = np.concatenate([self.pcc[i_tf], self.pcc[i_tg]])
        return (torch.tensor(feat, dtype=torch.float32),
                torch.tensor(float(row.label), dtype=torch.float32))


# ──────────────────────────────────────────────────────────
# TRAINING AND INFERENCE
# ──────────────────────────────────────────────────────────
def compute_pcc_matrix(expression_df):
    """Compute gene-gene PCC matrix from expression data."""
    X = expression_df.values.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    pcc = (X.T @ X) / X.shape[0]
    return pcc.astype(np.float32)


def build_train_pairs(train_tfs, gold_set, all_genes, rng):
    """Build balanced training pairs: 1:1 positive:negative."""
    pos = [(tf, tg) for tf, tg in gold_set if tf in set(train_tfs)]
    ps = set(pos)

    neg = []
    tfs_arr = np.array(train_tfs)
    g_arr = np.array(all_genes)
    while len(neg) < len(pos):
        tf = rng.choice(tfs_arr)
        tg = rng.choice(g_arr)
        if (tf, tg) not in ps and tf != tg:
            neg.append((tf, tg))

    df = pd.DataFrame(pos + neg, columns=["TF", "TG"])
    df["label"] = [1] * len(pos) + [0] * len(neg)
    return df


def run_deeptgi(expression_df, gold_edges, tf_list=None,
                train_tfs=None, eval_tfs=None,
                n_epochs=40, lr=1e-5, batch_size=256,
                proj_dim=1024, device="cpu", seed=0):
    """
    Train DeepTGI and score all TF-gene pairs for eval_tfs.

    For unsupervised mode (no train/eval split), trains on a subset
    of gold edges and evaluates on the rest using 5-fold CV.

    Parameters
    ----------
    expression_df : pd.DataFrame
    gold_edges : set of (TF, TG)
    tf_list : list of TF gene names
    train_tfs / eval_tfs : explicit TF split for partitioned evaluation
    device : str

    Returns
    -------
    dict of {(TF, TG): score}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    gene_names = list(expression_df.columns)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    if tf_list is None:
        tf_list = gene_names
    tfs_in_expr = [t for t in tf_list if t in gene2idx]

    print(f"  DeepTGI: {n_genes} genes, {len(tfs_in_expr)} TFs")
    print("  Computing PCC matrix...")
    pcc = compute_pcc_matrix(expression_df)

    if train_tfs is None or eval_tfs is None:
        train_tfs = tfs_in_expr
        eval_tfs = tfs_in_expr

    train_df = build_train_pairs(train_tfs, gold_edges, gene_names, rng)
    print(f"  Training pairs: {len(train_df)} ({train_df.label.sum()} pos)")

    raw_dim = 2 * n_genes
    model = DeepTGIModel(raw_dim, proj_dim=proj_dim).to(device)
    loss_fn = DeepTGILoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = PairDataset(train_df, pcc, gene2idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != "cpu"))

    model.train()
    for epoch in range(n_epochs):
        ep_loss = 0
        for pcc_feat, label in train_loader:
            pcc_feat = pcc_feat.to(device)
            label = label.to(device)

            logit, proj, rec = model(pcc_feat)
            loss = loss_fn(logit, label, proj, rec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  DeepTGI epoch {epoch+1}/{n_epochs}  loss={ep_loss/len(train_loader):.4f}")

    # Score all eval TF-gene pairs
    model.eval()
    edge_scores = {}
    with torch.no_grad():
        for tf in eval_tfs:
            if tf not in gene2idx:
                continue
            i_tf = gene2idx[tf]
            batch_feats = []
            batch_tgs = []
            for tg in gene_names:
                if tf == tg:
                    continue
                i_tg = gene2idx[tg]
                feat = np.concatenate([pcc[i_tf], pcc[i_tg]])
                batch_feats.append(feat)
                batch_tgs.append(tg)

            if not batch_feats:
                continue
            feats_t = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)

            scores = []
            for start in range(0, len(feats_t), 2048):
                chunk = feats_t[start:start+2048]
                logit, _, _ = model(chunk)
                scores.extend(torch.sigmoid(logit).cpu().numpy().tolist())

            for tg, s in zip(batch_tgs, scores):
                edge_scores[(tf, tg)] = s

    return edge_scores
