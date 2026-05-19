import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=101):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

class SpectralParamTwoTowerBalanced(nn.Module):
    def __init__(self, seq_len=101, d_model=128, nhead=4, num_layers=5,
                 dim_feedforward=256, dropout=0.1, proj_dim=128):
        super().__init__()
        # Энкодер спектра (без изменений)
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.spec_pool = nn.AdaptiveAvgPool1d(1)
        self.spec_head = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        # Раздельные энкодеры для каждого параметра
        self.param_a_encoder = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),nn.Dropout(0.1),
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, proj_dim)
        )
        self.param_b_encoder = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),nn.Dropout(0.1),
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, proj_dim)
        )
        
        # Важно: конкатенируем и нормализуем вместе
        self.param_bn = nn.BatchNorm1d(proj_dim * 2)  # 2 * proj_dim
        self.param_proj = nn.Linear(proj_dim * 2, proj_dim)  # сжатие обратно

    def forward(self, S, a, b):
        # Спектр
        x = self.input_proj(S.unsqueeze(-1))
        x = self.pos_encoder(x)
        x = self.encoder(x)
        z_spec = self.spec_pool(x.transpose(1,2)).squeeze(-1)
        z_spec = self.spec_head(z_spec)

        # Каждый параметр кодируется независимо
        z_a = self.param_a_encoder(a.unsqueeze(-1))
        z_b = self.param_b_encoder(b.unsqueeze(-1))
        
        # Конкатенация и нормализация
        z_param = torch.cat([z_a, z_b], dim=-1)
        z_param = self.param_bn(z_param)
        z_param = self.param_proj(z_param)  # (B, proj_dim)

        return z_spec, z_param, z_a, z_b