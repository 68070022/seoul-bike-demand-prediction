"""MLP architecture for bike demand regression."""
from torch import nn

class BikeMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)
