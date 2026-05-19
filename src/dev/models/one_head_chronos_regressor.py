"""
Model for predict msw
"""

import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=12, dropout=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Проекция на размерность для attention
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, input_dim]

        # Проецируем и добавляем размерность последовательности
        h = self.input_proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]

        # Создаем запрос
        query = self.query_proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]

        # Self-attention
        attended, _ = self.multihead_attn(query, h, h)

        # Residual + norm
        out = self.norm(h + attended)
        out = self.dropout(out)

        return out.squeeze(1)  # [batch, hidden_dim]

class ChronosRegressor(nn.Module):
    def __init__(self, encoder, tokenizer, output_dim=1):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        # self.head = nn.Linear(encoder.config.d_model, output_dim)
                # Слои с вниманием
        layers = []
        prev_dim = 512
        hidden_dims=[512 ,256, 128]
        num_heads=16
        dropout=0.2
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # Первый слой - attention
                layers.append(AttentionHead(prev_dim, hidden_dim, num_heads, dropout))
            else:
                # Последующие слои - обычные линейные
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                ])
            prev_dim = hidden_dim

        # Выходной слой
        layers.append(nn.Linear(prev_dim, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x приходит на GPU, но токенизатор требует CPU
        x_cpu = x.cpu()
        tokens = self.tokenizer.context_input_transform(x_cpu)
        token_ids = tokens[0] if isinstance(tokens, tuple) else tokens

        # Переносим token_ids на то же устройство, что и энкодер
        token_ids = token_ids.to(self.encoder.device)

        enc_out = self.encoder(input_ids=token_ids).last_hidden_state
        return self.head(enc_out[:, -1, :])