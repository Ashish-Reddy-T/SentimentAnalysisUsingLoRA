import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha/rank

        self.loraA = nn.Linear(in_features, rank, bias=False) 
        self.loraB = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.loraA.weight, a=5**0.5)
        nn.init.zeros_(self.loraB.weight)

    def forward(self, x):
        delta = self.loraB(self.loraA(x))   # (x*A)*B -->  ((B, S, D) * (B, D, R)) * (B, R, D) --> (B, S, R) * (B, R, D) --> (B, S, D)
        x = self.scaling * delta
        return x