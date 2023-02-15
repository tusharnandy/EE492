import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BlackBox(nn.Module):
    def __init__(self, name, initial_feature_dim, output_feature_dim):
        super(BlackBox, self).__init__()
        self.name = name
        if name == 'Logistic':
            self.net = nn.Linear(initial_feature_dim, output_feature_dim)

    def forward(self, x):
        if self.name == 'Logistic':
            return torch.sigmoid(self.net(x))


        