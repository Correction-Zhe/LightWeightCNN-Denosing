import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDnCNN(nn.Module):
    def __init__(self, depth=7, n_channels=32, image_channels=1, kernel_size=1):
        super(SimpleDnCNN, self).__init__()
        padding = kernel_size // 2
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        denoised = x - noise
        return denoised, noise