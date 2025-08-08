import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => ReLU => BN) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool followed by double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling and concatenation with skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed (due to odd input sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to get 1-channel output"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)
    

class UNet2p5D(nn.Module):
    """
    A 2.5D U-Net that processes local slice stacks with flexible conditioning that supports:
        - Missing/corrupted slice handling
        - Optional 3D volume-aware input encoding

    Args:
        in_channels (int): Number of input slices in the stack (e.g., 5).
        out_channels (int): Number of output channels (typically 1 for grayscale prediction).
        features (list of int): Number of channels at each U-Net depth.
        dropout_rate (float): Dropout probability. If 0.0, no dropout is applied.
        enable_3d_input (bool): If True, uses initial 3D preprocessing.
    """
    def __init__(self, in_channels=5, out_channels=1, features=None, dropout_rate=0.0, enable_3d_input=False):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.dropout_rate = dropout_rate
        self.enable_3d_input = enable_3d_input
        self.in_channels = in_channels

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if self.enable_3d_input:
            self.initial_3d = nn.Conv3d(1, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        if self.enable_3d_input:
            # Apply 3D convolution to the input stack to extract local volumetric context.
            # Input: (B, S, H, W) → Unsqueeze → (B, 1, S, H, W)
            x = x.unsqueeze(1)
            x = self.initial_3d(x)  # (B, C, S, H, W)
            x = x.mean(dim=2)       # Average across depth to return to 2D shape: (B, C, H, W)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.dropout(x4)  # bottleneck dropout
        x = self.up1(x4, x3)
        x = self.dropout(x)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
