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
    def __init__(self, x1_in_ch, x2_in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(x1_in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(x2_in_ch + out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # x1 is decoder input
        # Pad if needed
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
    

class ModulatedInputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.scale_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels * out_channels),
            nn.ReLU()
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, validity_mask):
        """
        x: [B, in_channels, H, W]
        validity_mask: [B, in_channels]
        """
        B, C, H, W = x.shape

        # Predict scaling matrix S for each sample in the batch
        S = self.scale_fc(validity_mask)  # Shape: [B, in_channels * out_channels]
        S = S.view(B, self.out_channels, self.in_channels)  # [B, C_out, C_in]

        # Apply scaled convolution (one per sample)
        weight = self.base_conv.weight  # [C_out, C_in, k, k]
        out = []
        for b in range(B):
            scaled_weight = S[b].unsqueeze(-1).unsqueeze(-1) * weight  # [C_out, C_in, k, k]
            out.append(F.conv2d(x[b].unsqueeze(0), scaled_weight, padding=1))

        return F.relu(torch.cat(out, dim=0))  # [B, C_out, H, W]


class UNet3DHybrid_DynamicFusion(nn.Module):
    """
    A hybrid 3D-2D U-Net variant for slice-wise inpainting using both volumetric context and adaptive modulation.

    This architecture integrates two main augmentations over a vanilla 2.5D U-Net:
        A 3D encoder applied to B-scan stacks to capture inter-slice spatial correlations.
        A dynamic modulation pathway that learns a sample-specific fusion of input features 
          based on slice validity using a learned projection.

    Args:
        in_channels (int): Number of slices per input stack (depth, S).
        out_channels (int): Number of output channels (typically 1 for grayscale prediction).
        features (list of int): Channel sizes at each encoder depth.
        dropout_rate (float): Dropout rate applied after each decoder stage.
        stack_size (int): Number of slices in the input stack (also defines depth of 3D conv).
    """
    def __init__(self, in_channels=1, out_channels=1, features=None, dropout_rate=0.0, stack_size=9):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.stack_size = stack_size

        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(features[0])
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))

        self.encoder2 = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(features[1])
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))

        self.encoder3 = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(features[2])
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2))

        self.axial_collapse = nn.Conv3d(features[2], features[2], kernel_size=(stack_size, 1, 1))

        self.up1 = Up(features[2], features[2], features[1])
        self.up2 = Up(features[1], features[1], features[0])
        self.up3 = Up(features[0], features[0], features[0] // 2)
        self.outc = OutConv(features[0] // 2, out_channels)

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, S, H, W), where S is the number of slices in the stack.
            validity_mask (Tensor or None): Tensor of shape (B, S) indicating which slices are valid (1) or missing (0).

        Returns:
            Tensor of shape (B, out_channels, H, W): Predicted reconstruction of the central slice.
        """
        # x: (B, S, H, W)
        B, S, H, W = x.shape
        x_3D = x.unsqueeze(1)  # (B, 1, S, H, W)

        # 3D Encoding
        x1 = self.encoder1(x_3D)  # (B, C1, S, H, W)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        x3_pooled = self.pool3(x3)
        x_bottleneck = self.axial_collapse(x3_pooled).squeeze(2)  # (B, C3, H, W)
        x = self.dropout(x_bottleneck)

        # Decoder
        x = self.up1(x, x3[:, :, S // 2])
        x = self.dropout(x)
        x = self.up2(x, x2[:, :, S // 2])
        x = self.dropout(x)
        x = self.up3(x, x1[:, :, S // 2])
        x = self.dropout(x)
        return self.outc(x)
