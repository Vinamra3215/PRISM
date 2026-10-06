import torch
import torch.nn as nn

# =========================================================================
# Architecture A: Concat + MLP (Baseline)
# =========================================================================
class ConcatMLP(nn.Module):
    def __init__(self, kronos_dim=768, stock_dim=10, market_dim=4, 
                 h1_dim=256, h2_dim=64, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(kronos_dim)
        in_dim = kronos_dim + stock_dim + market_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2_dim, 1)
        )
        
    def forward(self, kronos_emb, stock_feat, market_feat):
        k_norm = self.layer_norm(kronos_emb)
        x = torch.cat([k_norm, stock_feat, market_feat], dim=-1)
        return self.net(x).squeeze(-1)


# =========================================================================
# Architecture B: Cross-Attention Gating (MASTER-Style)
# =========================================================================
class CrossAttentionGating(nn.Module):
    def __init__(self, kronos_dim=768, stock_dim=10, num_market_tokens=4, 
                 d_model=256, n_heads=4, dropout_rate=0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(kronos_dim)
        
        # 1. Project stock representations into d_model
        self.proj_kronos = nn.Linear(kronos_dim, d_model)
        self.proj_stock = nn.Linear(stock_dim, d_model)
        self.proj_query = nn.Linear(2 * d_model, d_model)
        
        # 2. Project market signals into key/value tokens
        self.proj_market = nn.Linear(1, d_model)
        
        # 3. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )
        
        # 4. Gating Head: takes Q [256] + Attn [256] = 512 -> outputs gate [256]
        self.gate_dense = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        
        # 5. Output Prediction Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
    def forward(self, kronos_emb, stock_feat, market_feat):
        k_norm = self.layer_norm(kronos_emb)
        k_proj = self.proj_kronos(k_norm)                # [B, 256]
        s_proj = self.proj_stock(stock_feat)             # [B, 256]
        
        stock_repr = torch.cat([k_proj, s_proj], dim=-1) # [B, 512]
        Q = self.proj_query(stock_repr)                  # [B, 256]
        
        market_tokens = market_feat.unsqueeze(-1)        # [B, 4, 1]
        KV = self.proj_market(market_tokens)            # [B, 4, 256]
        
        # Cross-Attention: Q attends to market tokens KV
        attn_out, _ = self.cross_attn(query=Q.unsqueeze(1), key=KV, value=KV)
        attn_out = attn_out.squeeze(1)                  # [B, 256]
        
        # Gate input: Q (256) + attn_out (256) = 512
        gate_input = torch.cat([Q, attn_out], dim=-1)   # [B, 512]
        gate = self.sigmoid(self.gate_dense(gate_input)) # [B, 256]
        
        # Convex blend: (1 - g) * Q + g * attn_out
        gated_repr = gate * attn_out + (1.0 - gate) * Q  # [B, 256]
        
        return self.head(gated_repr).squeeze(-1)         # [B]


# =========================================================================
# Architecture C: FiLM Conditioning (Feature-wise Linear Modulation)
# =========================================================================
class FiLMConditioning(nn.Module):
    def __init__(self, kronos_dim=768, stock_dim=10, market_dim=4, 
                 film_hidden_dim=64, h1_dim=256, h2_dim=64, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(kronos_dim)
        
        # FiLM Generator: predicts scale (gamma) and shift (beta)
        self.film_backbone = nn.Sequential(
            nn.Linear(market_dim, film_hidden_dim),
            nn.ReLU()
        )
        self.to_gamma = nn.Linear(film_hidden_dim, kronos_dim)
        self.to_beta = nn.Linear(film_hidden_dim, kronos_dim)
        
        # Prediction Network
        in_dim = kronos_dim + stock_dim
        self.pred_net = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2_dim, 1)
        )
        
    def forward(self, kronos_emb, stock_feat, market_feat):
        k_norm = self.layer_norm(kronos_emb)
        
        # Compute dynamic gamma and beta from market regime
        film_h = self.film_backbone(market_feat)
        gamma = self.to_gamma(film_h)
        beta = self.to_beta(film_h)
        
        # Modulate Kronos temporal state
        modulated_kronos = gamma * k_norm + beta
        
        combined = torch.cat([modulated_kronos, stock_feat], dim=-1)
        return self.pred_net(combined).squeeze(-1)
