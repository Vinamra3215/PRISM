"""The three requested Block 3 PyTorch architectures."""

from __future__ import annotations

import torch
from torch import nn


class ConcatMLP(nn.Module):
    def __init__(self, embedding_dim, stock_dim, market_dim, hidden1=256, hidden2=64, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + stock_dim + market_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden2, 1),
        )

    def forward(self, embedding, stock, market, return_aux=False):
        prediction = self.network(torch.cat([embedding, stock, market], dim=-1)).squeeze(-1)
        return (prediction, {}) if return_aux else prediction


class CrossAttentionGating(nn.Module):
    def __init__(self, embedding_dim, stock_dim, market_dim, d_model=128, n_heads=4,
                 head_hidden=32, dropout=0.2):
        super().__init__()
        self.kronos_projection = nn.Linear(embedding_dim, d_model)
        self.feature_projection = nn.Linear(stock_dim, d_model)
        self.query_projection = nn.Linear(2 * d_model, d_model)
        self.market_projection = nn.Linear(1, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.prediction = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1)
        )
        self.market_dim = market_dim

    def forward(self, embedding, stock, market, return_aux=False):
        stock_repr = torch.cat(
            [self.kronos_projection(embedding), self.feature_projection(stock)], dim=-1
        )
        query = self.query_projection(stock_repr).unsqueeze(1)
        market_tokens = self.market_projection(market.unsqueeze(-1))
        attended, weights = self.attention(
            query, market_tokens, market_tokens, need_weights=True, average_attn_weights=False
        )
        attended = attended.squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([query.squeeze(1), attended], dim=-1)))
        gated = gate * attended + (1.0 - gate) * query.squeeze(1)
        prediction = self.prediction(gated).squeeze(-1)
        auxiliary = {"attention": weights.squeeze(2), "gate": gate}
        return (prediction, auxiliary) if return_aux else prediction


class FiLMConditioning(nn.Module):
    def __init__(self, embedding_dim, stock_dim, market_dim, film_hidden=64,
                 hidden1=256, hidden2=64, dropout=0.2):
        super().__init__()
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, film_hidden), nn.ReLU())
        self.gamma = nn.Linear(film_hidden, embedding_dim)
        self.beta = nn.Linear(film_hidden, embedding_dim)
        self.prediction = nn.Sequential(
            nn.Linear(embedding_dim + stock_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden2, 1),
        )

    def forward(self, embedding, stock, market, return_aux=False, mode="full"):
        hidden = self.market_encoder(market)
        gamma = self.gamma(hidden)
        beta = self.beta(hidden)
        if mode == "gamma_only":
            modulated = gamma * embedding
        elif mode == "beta_only":
            modulated = embedding + beta
        elif mode == "none":
            modulated = embedding
        else:
            modulated = gamma * embedding + beta
        prediction = self.prediction(torch.cat([modulated, stock], dim=-1)).squeeze(-1)
        auxiliary = {"gamma": gamma, "beta": beta}
        return (prediction, auxiliary) if return_aux else prediction


def build_model(name, dimensions, params):
    common = (dimensions["embedding"], dimensions["stock"], dimensions["market"])
    if name == "concat":
        return ConcatMLP(*common, hidden1=params["hidden1"], hidden2=params["hidden2"],
                         dropout=params["dropout"])
    if name == "attention":
        return CrossAttentionGating(*common, d_model=params["d_model"], n_heads=params["n_heads"],
                                    head_hidden=params["head_hidden"], dropout=params["dropout"])
    if name == "film":
        return FiLMConditioning(*common, film_hidden=params["film_hidden"],
                                hidden1=params["hidden1"], hidden2=params["hidden2"],
                                dropout=params["dropout"])
    raise ValueError(f"Unknown architecture: {name}")
