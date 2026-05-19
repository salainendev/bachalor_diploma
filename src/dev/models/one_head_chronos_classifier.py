"""
model for predict h2a (bad result)
"""
import torch
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

class ChronosClassifier(nn.Module):
    """
    Бинарный классификатор на основе энкодера Chronos (T5).
    Выход: вероятность класса 1 (после сигмоиды)
    """
    def __init__(self, encoder, tokenizer, output_dim=1):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        
        # Замораживаем энкодер
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Слои с вниманием
        layers = []
        prev_dim = 512  # размерность эмбеддингов Chronos-T5-small
        hidden_dims = [512, 256, 128]
        num_heads = 16
        dropout = 0.2
        
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

        # Выходной слой (1 нейрон, без активации - логит)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Сигмоида для получения вероятности
        self.head = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: тензор [batch, 101] (временной ряд)
        
        Returns:
            probabilities: [batch] — вероятность класса 1
        """
        # Токенизация (требует CPU)
        x_cpu = x.cpu()
        tokens = self.tokenizer.context_input_transform(x_cpu)
        token_ids = tokens[0] if isinstance(tokens, tuple) else tokens
        
        # Переносим token_ids на устройство энкодера
        token_ids = token_ids.to(self.encoder.device)
        
        # Энкодинг
        enc_out = self.encoder(input_ids=token_ids).last_hidden_state  # [batch, seq_len, 512]
        
        # Берём последний токен как представление всей последовательности
        embedding = enc_out[:, -1, :]  # [batch, 512]
        
        # Проходим через голову (логиты)
        logits = self.head(embedding)  # [batch, 1]
        
        # Вероятность через сигмоиду
        proba = self.sigmoid(logits)
        
        return proba.squeeze(1)  # [batch]
    
    def predict(self, x, threshold=0.5):
        """
        Предсказание класса (0 или 1).
        
        Args:
            x: входной тензор [batch, 101]
            threshold: порог классификации (по умолчанию 0.5)
        
        Returns:
            predictions: [batch] — предсказанные классы
        """
        probs = self.forward(x)
        return (probs >= threshold).int()
    
    def predict_proba(self, x):
        """
        Возвращает вероятности класса 1.
        """
        return self.forward(x)
    
    def predict_with_logits(self, x):
        """
        Возвращает (логиты, вероятности, предсказания).
        """
        x_cpu = x.cpu()
        tokens = self.tokenizer.context_input_transform(x_cpu)
        token_ids = tokens[0] if isinstance(tokens, tuple) else tokens
        token_ids = token_ids.to(self.encoder.device)
        
        enc_out = self.encoder(input_ids=token_ids).last_hidden_state
        embedding = enc_out[:, -1, :]
        
        logits = self.head(embedding).squeeze(1)  # [batch]
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        
        return logits, probs, preds