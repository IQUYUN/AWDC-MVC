import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return F.softmax(self.head(x), dim=1) # 最好不要softmax，交叉熵自己会处理
