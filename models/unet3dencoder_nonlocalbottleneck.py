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
    

class Conv3DEncoder(nn.Module):
    """
    Encodes the full input stack with 3D convolutions.
    Projects the encoded volume down to a single 2D feature map by collapsing the slice dimension.

    Input: (B, S, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Conv3d(out_channels, out_channels, kernel_size=(in_channels, 1, 1))

    def forward(self, x):  # x: (B, S, H, W)
        x = x.unsqueeze(1)  # Add channel dim → (B, 1, S, H, W)
        x = self.encoder(x)
        x = self.project(x)  # (B, C, 1, H, W)
        x = x.squeeze(2)  # (B, C, H, W)
        return x

class NonLocalBlock(nn.Module):
    """
    Implements a non-local attention block over the 2D spatial domain.
    Captures long-range dependencies across the entire B-scan, encouraging vessel continuity and global structure awareness.
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
        B, C, H, W = x.shape
        N = H * W

        g_x = self.g(x).view(B, self.inter_ch, -1).permute(0, 2, 1)        # [B, N, C']
        theta_x = self.theta(x).view(B, self.inter_ch, -1).permute(0, 2, 1)  # [B, N, C']
        phi_x = self.phi(x).view(B, self.inter_ch, -1)                     # [B, C', N]

        f = torch.matmul(theta_x, phi_x)     # [B, N, N]
        f_div_C = F.softmax(f, dim=-1)  # Attention weights over all spatial positions

        y = torch.matmul(f_div_C, g_x)       # [B, N, C']
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_ch, H, W)  # [B, C', H, W]

        W_y = self.W(y)
        return W_y + x


class UNet3DEncoder_NonLocalBottleneck(nn.Module):
    """
    U-Net variant with a volumetric 3D encoder and a 2D non-local bottleneck.

    Key Differences from our baseline UNet2p5D:
    - Uses a fixed 3D encoder (Conv3DEncoder).
    - Entire 3D stack is projected to a 2D feature map before entering the decoder path.
    - A NonLocalBlock is applied at the bottleneck to capture long-range spatial dependencies
      and enhance contextual awareness — particularly useful for vascular continuity in projections.

    Args:
        in_channels (int): Number of input slices in the stack (e.g., 5).
        out_channels (int): Number of output channels (typically 1).
        features (list[int]): Feature sizes for each U-Net level.
        dropout_rate (float): Optional dropout at bottleneck and upsampling.
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
