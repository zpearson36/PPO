import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))

        return self.l3(x)
