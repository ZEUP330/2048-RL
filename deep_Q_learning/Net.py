import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, n_s, n_a):
        self.n_state = n_s
        self.n_action = n_a
        super(Net, self).__init__()
        self.full_convolution = nn.Sequential(OrderedDict([
            ('fc1', nn.Conv2d(1, 64, 5, 1, 2)),
            ('fc2', nn.Conv2d(64, 128, 5, 1, 2)),
            ('fc3', nn.Conv2d(128, 512, 5, 1, 2)),
            ('fc4', nn.Conv2d(512, 128, 5, 1, 2)),
            ('fc5', nn.Conv2d(128, 64, 5, 1, 2)),
            ('fc7', nn.Conv2d(64, 16, 5, 1, 2)),
            ('fc8', nn.Conv2d(16, n_a, 5, n_s, 2)),
        ]))

        self.full_connect = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.n_state, 64)),
            ('fc2', nn.Linear(64, 64)),
            ('fc3', nn.Linear(64, self.n_action)),
        ]))

    def forward(self, x):
        return self.full_convolution(x).reshape(-1, self.n_action)
        # return self.full_connect(x.reshape(-1, self.n_state))
