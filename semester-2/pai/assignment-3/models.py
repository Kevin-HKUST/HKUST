import torch
import torch.nn as nn

class ColorizationCnn(nn.Module):
    def __init__(self):
        super(ColorizationCnn, self).__init__()
        # Encoder 1: 1 -> 64
        self.enc1_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc1_pool = nn.MaxPool2d(2)
        # Encoder 2: 64 -> 128
        self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc2_pool = nn.MaxPool2d(2)
        # Bottleneck: 128 -> 256
        self.bot_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bot_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # Decoder 1: 256 -> 128
        self.dec1_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Decoder 2: 128 -> 64
        self.dec2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Output: 64 -> 32 -> 2
        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc1_conv1(x))
        x = self.relu(self.enc1_conv2(x))
        x = self.enc1_pool(x)
        x = self.relu(self.enc2_conv1(x))
        x = self.relu(self.enc2_conv2(x))
        x = self.enc2_pool(x)
        x = self.relu(self.bot_conv1(x))
        x = self.relu(self.bot_conv2(x))
        x = self.relu(self.dec1_conv(x))
        x = self.dec1_up(x)
        x = self.relu(self.dec2_conv(x))
        x = self.dec2_up(x)
        x = self.relu(self.out_conv1(x))
        x = self.out_conv2(x)
        return x