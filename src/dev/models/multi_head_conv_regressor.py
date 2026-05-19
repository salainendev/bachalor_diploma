"""
Model for predict He/H
"""

import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding='same', dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SqueezeExcitation(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act_out = nn.SiLU()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out)
        
        out += residual
        return self.act_out(out)

class AdvancedConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, output_dim=512, dropout=0.2):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block1 = ResidualConvBlock(64, 128, kernel_size=3, dilation=1, dropout=dropout)
        
        self.down1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.SiLU()
        )

        self.block2 = ResidualConvBlock(128, 256, kernel_size=3, dilation=2, dropout=dropout)
        
        self.down2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )

        self.block3 = ResidualConvBlock(256, 512, kernel_size=3, dilation=4, dropout=dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(25) 
        
        flatten_dim = 25 * 512
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim), 
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        
        x = self.block2(x)
        x = self.down2(x)
        
        x = self.block3(x)
        
        x = self.global_pool(x)
        
        x = self.head(x)
        return x


class MultiHeadConv1DRegressor(nn.Module):
    def __init__(self, input_channels=1, output_dim=1, num_heads=5,
                 hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        feat_dim = hidden_dims[0]
        self.feature_extractor = AdvancedConvFeatureExtractor(
            in_channels=input_channels,
            output_dim=feat_dim,
            dropout=dropout
        )

        layers = []
        prev_dim = feat_dim
        
        for hidden_dim in hidden_dims[1:]:
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
        """
        x: тензор формы [batch, channels, 101] или [batch, 101].
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        features = self.feature_extractor(x)
        features = self.shared_features(features)


        preds = torch.stack([head(features) for head in self.pred_heads], dim=1)    

        return preds