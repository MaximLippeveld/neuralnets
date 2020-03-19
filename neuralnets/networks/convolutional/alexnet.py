import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, image_shape, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(image_shape[0], 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2), #kernel_size
            nn.Conv2d(64, 192, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        tmp = torch.empty([1] + list(image_shape))
        flat = self.features(tmp)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flat.shape[1], 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x