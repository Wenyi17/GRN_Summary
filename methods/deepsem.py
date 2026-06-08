"""
DeepSEM (Shu et al., 2021)

Faithful reimplementation of the DeepSEM architecture:
Gaussian-Mixture VAE with a learnable structural equation adj_A matrix.
Training alternates between MLP parameters and adj_A.

Architecture follows the original code (github.com/HantaoShu/DeepSEM):
- InferenceNet: qyx (GumbelSoftmax) and qzyx (Gaussian)
- GenerativeNet: pzy and pxz
- adj_A parameterizes the causal DAG via (I - adj_A^T)
- Loss = reconstruction (MSE) + gaussian_loss + categorical entropy + L1 sparsity
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset


def _kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class _LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise ValueError(f"Unknown rec_type: {rec_type}")
        return loss

    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        log2pi = torch.log(torch.tensor([2.0 * np.pi], device=x.device))
        return -0.5 * torch.mean(
            log2pi.sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class _GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim):
        super().__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.device)
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax_sample(logits, temperature)
        return logits, prob, y


class _Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)


class _InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super().__init__()
        self.inference_qyx = nn.ModuleList([
            nn.Linear(n_gene, z_dim), nonLinear,
            nn.Linear(z_dim, z_dim), nonLinear,
            _GumbelSoftmax(z_dim, y_dim)
        ])
        self.inference_qzyx = nn.ModuleList([
            nn.Linear(x_dim + y_dim, z_dim), nonLinear,
            nn.Linear(z_dim, z_dim), nonLinear,
            _Gaussian(z_dim, 1)
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        return mu + noise * std

    def qyx(self, x, temperature):
        for i, layer in enumerate(self.inference_qyx):
            if i == len(self.inference_qyx) - 1:
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def qzxy(self, x, y):
        concat = torch.cat((x, y.unsqueeze(1).repeat(1, x.shape[1], 1)), dim=2)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, adj, temperature=1.0):
        logits, prob, y = self.qyx(x.squeeze(2), temperature)
        mu, logvar = self.qzxy(x, y)
        mu_ori = mu
        mu = torch.matmul(mu, adj)
        logvar = torch.matmul(logvar, adj)
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        return {
            'mean': mu, 'var': var, 'gaussian': z,
            'logits': logits, 'prob_cat': prob, 'categorical': y, 'mu_ori': mu_ori,
        }


class _GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super().__init__()
        self.n_gene = n_gene
        self.y_mu = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.y_var = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.generative_pxz = nn.ModuleList([
            nn.Linear(1, z_dim), nonLinear,
            nn.Linear(z_dim, z_dim), nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pzy(self, y):
        return self.y_mu(y), self.y_var(y)

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y, adj):
        y_mu, y_logvar = self.pzy(y)
        y_mu = torch.matmul(y_mu, adj)
        y_logvar = torch.matmul(y_logvar, adj)
        y_var = torch.exp(y_logvar)
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        return {
            'y_mean': y_mu.view(-1, self.n_gene),
            'y_var': y_var.view(-1, self.n_gene),
            'x_rec': x_rec,
        }


class _VAE_EAD(nn.Module):
    """Device-agnostic reimplementation of VAE_EAD from DeepSEM."""

    def __init__(self, adj_A, x_dim, z_dim, y_dim):
        super().__init__()
        self.adj_A = nn.Parameter(
            torch.from_numpy(adj_A).double(), requires_grad=True
        )
        self.n_gene = n_gene = len(adj_A)
        nonLinear = nn.Tanh()
        self.inference = _InferenceNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.generative = _GenerativeNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.losses = _LossFunctions()

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _one_minus_A_t(self, adj):
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        return eye - adj.transpose(0, 1)

    def forward(self, x, dropout_mask, temperature=1.0, opt=None):
        x_ori = x
        x = x.view(x.size(0), -1, 1)
        mask = (torch.ones(self.n_gene, self.n_gene, device=x.device) -
                torch.eye(self.n_gene, device=x.device)).float()
        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)
        out_inf = self.inference(x, adj_A_t, temperature)
        z, y = out_inf['gaussian'], out_inf['categorical']
        z_inv = torch.matmul(z, adj_A_t_inv)
        out_gen = self.generative(z_inv, y, adj_A_t)
        output = {**out_inf, **out_gen}
        dec = output['x_rec']
        loss_rec = self.losses.reconstruction_loss(x_ori, dec, dropout_mask, 'mse')
        loss_gauss = self.losses.gaussian_loss(
            z, output['mean'], output['var'],
            output['y_mean'], output['y_var'],
        ) * opt.beta
        loss_cat = (-self.losses.entropy(output['logits'], output['prob_cat'])
                    - np.log(0.1)) * opt.beta
        loss = loss_rec + loss_gauss + loss_cat
        return loss, loss_rec, loss_gauss, loss_cat, dec, y, output['mean']


class _Opt:
    def __init__(self, **kwargs):
        self.n_epochs = kwargs.get("n_epochs", 120)
        self.batch_size = kwargs.get("batch_size", 64)
        self.alpha = kwargs.get("alpha", 100)
        self.beta = kwargs.get("beta", 1)
        self.lr = kwargs.get("lr", 1e-4)
        self.gamma = kwargs.get("gamma", 0.95)
        self.n_hidden = kwargs.get("n_hidden", 128)
        self.K = kwargs.get("K", 1)
        self.K1 = kwargs.get("K1", 1)
        self.K2 = kwargs.get("K2", 2)


def run_deepsem(expression_df, tf_list=None, n_epochs=120, lr=1e-4,
                hidden_dim=128, device="cpu", seed=0):
    """
    Run DeepSEM and return edge scores from the learned adj_A.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Samples as rows, genes as columns.
    tf_list : list or None
    n_epochs : int
    lr : float
    hidden_dim : int
    device : str ('cpu' or 'cuda')
    seed : int

    Returns
    -------
    dict of {(TF, TG): abs(adj_A_score)}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gene_names = list(expression_df.columns)
    n_genes = len(gene_names)
    tf_set = set(tf_list) & set(gene_names) if tf_list is not None else set(gene_names)

    data_values = expression_df.values.astype(np.float32)
    data_values = (data_values - data_values.mean(0)) / (data_values.std(0) + 1e-8)

    opt = _Opt(n_epochs=n_epochs, lr=lr, n_hidden=hidden_dim,
               batch_size=min(64, len(data_values)))

    adj_A_init = np.ones([n_genes, n_genes]) / (n_genes - 1) + \
        (np.random.rand(n_genes * n_genes) * 0.0002).reshape([n_genes, n_genes])
    np.fill_diagonal(adj_A_init, 0)

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    vae = _VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().to(dev)

    feat = torch.FloatTensor(data_values)
    dataset = TensorDataset(feat, torch.LongTensor(list(range(len(feat)))))
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.RMSprop(vae.parameters(), lr=opt.lr)
    optimizer2 = torch.optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.gamma)

    vae.train()
    for epoch in range(opt.n_epochs + 1):
        loss_all = []
        vae.adj_A.requires_grad = (epoch % (opt.K1 + opt.K2) >= opt.K1)

        for data_batch in dataloader:
            inputs, _ = data_batch
            inputs = inputs.to(dev)
            optimizer.zero_grad()
            temperature = max(0.95 ** epoch, 0.5)
            loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(
                inputs, dropout_mask=None, temperature=temperature, opt=opt,
            )
            sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
            loss = loss + sparse_loss
            loss.backward()
            loss_all.append(loss.item())

            if epoch % (opt.K1 + opt.K2) < opt.K1:
                optimizer.step()
            else:
                optimizer2.step()

        scheduler.step()

        if (epoch + 1) % 30 == 0:
            print(f"  DeepSEM epoch {epoch+1}/{opt.n_epochs}  "
                  f"loss={np.mean(loss_all):.4f}")

    adj_A = vae.adj_A.cpu().detach().numpy()

    edge_scores = {}
    for j in range(n_genes):
        tf = gene_names[j]
        if tf not in tf_set:
            continue
        for i in range(n_genes):
            if i == j:
                continue
            edge_scores[(tf, gene_names[i])] = abs(float(adj_A[i, j]))

    return edge_scores
