import torch.nn as nn
import torch
from . import core

class GlobalPoolConcat(nn.Module):

    def __init__(self, layer_shape):
        super(GlobalPoolConcat, self).__init__()
        self.avg = nn.AvgPool2d(layer_shape[2:])
        self.max = nn.MaxPool2d(layer_shape[2:])

    def forward(self, x):
        return torch.cat([
            torch.squeeze(self.avg(x)),
            torch.squeeze(self.max(x))
        ], dim=1)


class VGGDrop(nn.Module):
    """
    https://github.com/OATML/bdl-benchmarks/blob/alpha/baselines/diabetic_retinopathy_diagnosis/mc_dropout/model.py
    """

    def __init__(self, input_shape, num_filters, num_classes, dropout_rate):
        super(VGGDrop, self).__init__()

        self.layers = nn.ModuleList()
        act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.layers.extend([
            # block 1
            nn.Conv2d(input_shape[0], num_filters, 3, 2, 1),
            nn.BatchNorm2d(num_filters),
            act,
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(3, 2),
            # block 2
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_filters),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_filters),
            act,
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(3, 1),
            # block 3
            nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            nn.BatchNorm2d(num_filters*2),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*2, num_filters*2, 3, 1, 1),
            nn.BatchNorm2d(num_filters*2),
            act,
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(3, 1),
            # block 4
            nn.Conv2d(num_filters*2, num_filters*4, 3, 1, 1),
            nn.BatchNorm2d(num_filters*4),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
            nn.BatchNorm2d(num_filters*4),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
            nn.BatchNorm2d(num_filters*4),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
            nn.BatchNorm2d(num_filters*4),
            act,
            # block 5
            nn.Conv2d(num_filters*4, num_filters*8, 3, 1, 1),
            nn.BatchNorm2d(num_filters*8),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
            nn.BatchNorm2d(num_filters*8),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
            nn.BatchNorm2d(num_filters*8),
            act,
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
            nn.BatchNorm2d(num_filters*8),
            act
        ])

        # get final shape for global avg pooling
        tmp = torch.zeros([1] + input_shape)
        for layer in self.layers:
            tmp = layer(tmp)

        self.layers.extend([
            GlobalPoolConcat(tmp.shape),
            nn.Linear(num_filters * 8 * 2, num_classes)
        ])

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
