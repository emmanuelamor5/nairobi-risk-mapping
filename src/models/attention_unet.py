"""
Attention U-Net architecture (Oktay et al., 2018), matching proposal
Sec "Theoretical Framework" and "Model Training": a standard U-Net
encoder-decoder with attention gates applied at each skip connection,
so the decoder learns to suppress irrelevant encoder features (background
clutter, non-flooded structures) rather than concatenating them unfiltered.

Input channels default to 9: the 6 Sentinel-2 optical bands plus elevation,
slope, and TWI from the raster stack (see src/data_pipeline.py). The
flood_label band is the training target, not a model input, and is
excluded here -- src/models/dataset.py handles that split when loading
tiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU -- the standard U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention gate (Oktay et al., 2018). Takes the decoder's upsampled
    "gating signal" (g) and the corresponding encoder skip connection (x),
    and produces a per-pixel attention coefficient in [0,1] that scales x
    before it's concatenated into the decoder path. This is the mechanism
    the proposal cites for suppressing background clutter (dense roofing,
    vegetation, shadow) and focusing on hydrologically relevant regions.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # g and s may differ in spatial size by a pixel due to odd input
        # dimensions propagating through pooling -- align before summing.
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary flood segmentation. Architecture: a 4-level
    encoder-decoder with attention gates at every skip connection.

    Channel widths (32, 64, 128, 256, 512 at the bottleneck) are a
    reasonably standard, GPU-memory-conscious choice for 256x256 tiles on a
    Colab T4 (per proposal Sec "Data Processing" tiling rationale) -- deep
    enough to be a meaningful transfer-learning target, shallow enough to
    train comfortably on the small Kibera MVP tile count without immediately
    overfitting a huge parameter count to only a handful of spatial samples.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16

        # Encoder (contracting path) -- this is the part transfer learning targets.
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c4, c5)

        # Decoder (expanding path) -- trained from random init per proposal's
        # transfer-learning plan, alongside the attention gates.
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=c4, skip_channels=c4, inter_channels=c4 // 2)
        self.dec4 = ConvBlock(c4 * 2, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=c3, skip_channels=c3, inter_channels=c3 // 2)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=c2, skip_channels=c2, inter_channels=c2 // 2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=c1, skip_channels=c1, inter_channels=c1 // 2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention-gated skip connections
        d4 = self.up4(b)
        e4_att = self.att4(gate=d4, skip=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(gate=d3, skip=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(gate=d2, skip=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(gate=d1, skip=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)  # raw logits -- apply sigmoid outside for probabilities

    def encoder_parameters(self):
        """Parameters belonging only to the encoder -- used to freeze/unfreeze for transfer learning."""
        modules = [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck]
        for m in modules:
            yield from m.parameters()

    def decoder_and_attention_parameters(self):
        """Parameters belonging to the decoder and attention gates -- what proposal Sec 'Model Training' trains from scratch."""
        modules = [
            self.up4, self.att4, self.dec4,
            self.up3, self.att3, self.dec3,
            self.up2, self.att2, self.dec2,
            self.up1, self.att1, self.dec1,
            self.out_conv,
        ]
        for m in modules:
            yield from m.parameters()
