"""
Model for predict Fxuv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=12, dropout=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        h = self.input_proj(x).unsqueeze(1)          
        query = self.query_proj(x).unsqueeze(1) 
        attended, _ = self.multihead_attn(query, h, h)
        out = self.norm(h + attended)
        out = self.dropout(out)
        return out.squeeze(1)  


class MultiHeadChronosRegressor(nn.Module):
    def __init__(self, encoder, tokenizer, output_dim=1, num_heads=5,
                 hidden_dims=[512, 256], num_attn_heads=16, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_heads = num_heads
        self.output_dim = output_dim

        input_dim = encoder.config.d_model
        layers = []

        layers.append(AttentionHead(input_dim, hidden_dims[0], num_attn_heads, dropout))
        prev_dim = hidden_dims[0]

        for i, hidden_dim in enumerate(hidden_dims[1:], start=1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared_features = nn.Sequential(*layers)

        self.pred_heads = nn.ModuleList([
            nn.Linear(prev_dim, self.output_dim) for _ in range(num_heads)
        ])

        

    def forward(self, x):
        
        x_cpu = x.cpu()
        tokens = self.tokenizer.context_input_transform(x_cpu)
        token_ids = tokens[0] if isinstance(tokens, tuple) else tokens
        token_ids = token_ids.to(self.encoder.device)
        enc_out = self.encoder(input_ids=token_ids).last_hidden_state
        emb = enc_out[:, -1, :]

        features = self.shared_features(emb)

        preds = torch.stack([head(features) for head in self.pred_heads], dim=1)
              
        return preds