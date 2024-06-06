""" Full assembly of the parts to form the complete network """

import torch
from cunet_parts import *


class CUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inconv = Conv_Block(in_channels, 8)
        self.down1 = Down_Block(8, 16)
        self.down2 = Down_Block(16, 32)
        self.down3 = Down_Block(32, 64)
        self.down4 = Down_Block(64, 128)

        # Decoder
        self.up4 = Upscale_Block(128)
        self.up4conv = Conv_Block(128, 64)

        self.up3 = Upscale_Block(64)
        self.up3conv = Conv_Block(64, 32)

        self.up2 = Upscale_Block(32)
        self.up2conv = Conv_Block(32, 16)

        self.up1 = Upscale_Block(16)
        self.up1conv = Conv_Block(16, 8)

        self.outconv = OutConv_Block(8, out_channels)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up4conv(x)

        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up3conv(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up2conv(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up1conv(x)

        x = self.outconv(x)
        return x
