import torch.nn as nn
import torch

class Ensemble(nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([m(x) for m in self.models])
        return torch.mean(outputs, axis=0)
