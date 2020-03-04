import torch.nn as nn
import torch

class DeepFlow(nn.Module):
    
    class _dual_layer(nn.Module):
    
        def __init__(self, f_in, f_out_1, f_out_2):
            super().__init__()
            self.left = nn.Sequential(
                nn.Conv2d(f_in, f_out_1, 1),
                nn.BatchNorm2d(f_out_1),
                nn.ReLU()
            )
            self.right = nn.Sequential(
                nn.Conv2d(f_in, f_out_2, 3, padding=1),
                nn.BatchNorm2d(f_out_2),
                nn.ReLU()
            )

        def forward(self, x):
            return torch.cat([
                self.left.forward(x), 
                self.right.forward(x)], 1)

    class _dual_downsample_layer(nn.Module):
    
        def __init__(self, f_in, f_out):
            super().__init__()

            self.layers = nn.Sequential(
                nn.Conv2d(f_in, f_out, 3, stride=2, padding=1),
                nn.BatchNorm2d(f_out),
                nn.ReLU()
            )
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            return torch.cat([
                self.layers.forward(x),
                self.pool(x)
            ], axis=1)

    def __init__(self, image_shape, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_shape[0], 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            self._dual_layer(96, 32, 32),
            self._dual_layer(64, 32, 48),
            self._dual_downsample_layer(80, 80),
            self._dual_layer(160, 112, 48),
            self._dual_layer(160, 96, 64),
            self._dual_layer(160, 80, 80),
            self._dual_layer(160, 48, 96),
            self._dual_downsample_layer(144, 96),
            self._dual_layer(240, 176, 160),
            self._dual_layer(336, 176, 160),
            self._dual_downsample_layer(336, 96),
            self._dual_layer(432, 176, 160),
            self._dual_layer(336, 176, 160),
            nn.Flatten()
        )

        tmp = torch.empty([1] + list(image_shape))
        flat = self.feature_extractor(tmp)

        self.classifier = nn.Sequential(
            nn.Linear(flat.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x
