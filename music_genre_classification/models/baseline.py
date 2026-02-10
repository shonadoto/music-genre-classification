import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, num_classes=10, n_mels=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        self.fc1 = nn.Linear(n_mels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(
                "BaselineModel expects input of shape\
                     [B, 1, n_mels, T] or [B, n_mels, T]"
            )

        mel_features = torch.mean(x, dim=2)

        x = self.fc1(mel_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
