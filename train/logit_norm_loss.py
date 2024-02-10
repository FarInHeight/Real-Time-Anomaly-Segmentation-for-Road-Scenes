import torch
from torch import nn
import torch.functional as F


class LogitNormLoss(nn.Module):

    def __init__(self, loss_func, t=0.04):
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.loss_func = loss_func

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return self.loss_func(logit_norm, target)
