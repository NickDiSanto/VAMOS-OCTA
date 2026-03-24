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
            DoubleConv(in_ch, out_ch),
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to get 1-channel output"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)


class Conv3DEncoder(nn.Module):
    """
    Encode the full slice stack with 3D convolutions and collapse depth to a 2D feature map.

    Input: (B, S, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv3d(out_channels, out_channels, kernel_size=(in_channels, 1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.project(x)
        x = x.squeeze(2)
        return x


class NonLocalBlock(nn.Module):
    """
    Non-local attention block over the 2D spatial domain.

    This block builds dense spatial attention to encourage long-range consistency
    across the reconstructed B-scan.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.inter_ch = in_channels // 2
        self.g = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.W = nn.Conv2d(self.inter_ch, in_channels, 1)

        nn.init.constant_(self.W.weight, 0)
        if self.W.bias is not None:
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Project to query/key/value embeddings and compute full spatial attention.
        g_x = self.g(x).view(batch_size, self.inter_ch, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_ch, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_ch, -1)

        attention = torch.matmul(theta_x, phi_x)
        attention = F.softmax(attention, dim=-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_ch, height, width)

        W_y = self.W(y)
        return W_y + x


class UNet3DEncoder_NonLocalBottleneck(nn.Module):
    """
    U-Net variant with a volumetric 3D encoder and a non-local bottleneck.

    Args:
        in_channels (int): Number of input slices in the stack.
        out_channels (int): Number of output channels.
        features (list[int]): Feature sizes for each U-Net level.
        dropout_rate (float): Optional dropout at the bottleneck.
    """

    def __init__(self, in_channels=5, out_channels=1, features=None, dropout_rate=0.0):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.volumetric_encoder = Conv3DEncoder(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.nonlocal_block = NonLocalBlock(features[3])
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        x1 = self.volumetric_encoder(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.nonlocal_block(x4)
        x4 = self.dropout(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
