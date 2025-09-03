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
    


class SpatialDepthInputBlockWithGating(nn.Module):
    """
    Applies 2D convolution to each slice in the stack independently, followed by
    1D convolution across the slice (depth) dimension. A gating mechanism, conditioned
    on the slice validity mask, modulates the final output to downweight unreliable context.

    Args:
        in_slices (int): Number of input slices in the stack.
        spatial_out (int): Number of output channels from 2D spatial conv per slice.
        depth_out (int): Number of output channels after depth-wise conv and gating.
    """
    def __init__(self, in_slices, spatial_out, depth_out):
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, spatial_out, kernel_size=3, padding=1)
        self.depth_conv = nn.Conv1d(spatial_out, depth_out, kernel_size=3, padding=1)
        self.gate_fc = nn.Sequential(
            nn.Linear(in_slices, depth_out),
            nn.Sigmoid()  # ensures multiplicative gating in [0, 1]
        )

    def forward(self, x, validity_mask):
        """
        Args:
            x (Tensor): Input tensor of shape (B, S, H, W)
            validity_mask (Tensor): Tensor of shape (B, S), where 1 indicates valid slices.

        Returns:
            Tensor of shape (B, depth_out, H, W): Processed spatial+depth-aware features.
        """
        B, S, H, W = x.shape

        # Apply 2D convolution independently to each slice
        x = x.view(B * S, 1, H, W)                     # (B*S, 1, H, W)
        x = F.relu(self.spatial_conv(x))              # (B*S, C_s, H, W)
        x = x.view(B, S, -1, H, W)                     # (B, S, C_s, H, W)

        # For each spatial location, apply 1D conv across stacked slices
        x = x.permute(0, 3, 4, 2, 1)                   # (B, H, W, C_s, S)
        x = x.reshape(-1, x.shape[3], x.shape[4])      # (B*H*W, C_s, S)
        x = self.depth_conv(x)                         # (B*H*W, C_d, S)
        x = x.mean(dim=2)                              # (B*H*W, C_d)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)     # (B, C_d, H, W)

        # Compute a gating signal conditioned on slice validity to control how much each feature channel should contribute
        gate = self.gate_fc(validity_mask)             # (B, C_d)
        gate = gate.view(B, -1, 1, 1)                  # (B, C_d, 1, 1)
        x = x * gate                                   # Apply gating per channel

        return x


class UNet2p5D_SeparableDepthGated(nn.Module):
    """
    A 2.5D U-Net variant that uses slice-wise spatial convolution followed by depth-wise 1D convolution to model depth continuity.

    Args:
        in_channels (int): Number of input slices in the stack (e.g., 5).
        out_channels (int): Number of output channels (typically 1 for grayscale prediction).
        features (list of int): Number of channels at each U-Net depth.
        dropout_rate (float): Dropout probability. If 0.0, no dropout is applied.
    """
    def __init__(self, in_channels=5, out_channels=1, features=None, dropout_rate=0.0):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.dropout_rate = dropout_rate

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")


        self.depth_conditioned_encoder = SpatialDepthInputBlockWithGating(
            in_slices=in_channels,
            spatial_out=features[0] // 2,
            depth_out=features[0]
        )

        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x, validity_mask):
        """
        Args:
            x (Tensor): Input volume of shape (B, S, H, W), where S is the number of slices per input stack.
            validity_mask (Tensor): Binary mask of shape (B, S) indicating valid (1) or missing (0) slices.

        Returns:
            Tensor of shape (B, 1, H, W): The predicted output slice.
        """
        x1 = self.depth_conditioned_encoder(x, validity_mask)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.dropout(x4)  # bottleneck dropout
        x = self.up1(x4, x3)
        x = self.dropout(x)    # decoder dropout (optional, same module)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
    
