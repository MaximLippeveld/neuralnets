import torch.nn as nn, torch

class ConvNet(nn.Module):

    def __init__(self, image_shape, num_classes, dropout_rate=0.1):
        super().__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(image_shape[0], 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        tmp = torch.empty([1] + list(image_shape))
        flat = self.extractor(tmp)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flat.shape[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)

        return x