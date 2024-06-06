""" Parts of the Complex U-Net model """

import torch.nn as nn

from complexLayers import ComplexConv2d as Conv2d
from complexLayers import ComplexBatchNorm2d as BatchNorm2d
from complexLayers import ComplexReLU as ReLU
from complexLayers import ComplexMaxPool2d as MaxPool2d
from complexLayers import (
    ComplexConvTranspose2d as ConvTranspose2d,
)


def Conv_Block(
    in_channels,
    out_channels,
    bn_layer=True,
):
    """(Conv => [BN] => ReLU)"""

    layers = []
    ####################################
    # First convolution layer
    layers.append(
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    )
    # BatchNorm layer
    if bn_layer:
        layers.append(BatchNorm2d(out_channels))
    # ReLU layer
    layers.append(ReLU())
    ####################################
    return nn.Sequential(*layers)

def Down_Block(in_channels, out_channels):
    """Downscaling with maxpool then double conv"""
    layers = []
    # MaxPool layer
    layers.append(MaxPool2d(kernel_size=2, stride=2))
    # DoubleConv layer
    layers.append(Conv_Block(in_channels, out_channels))
    return nn.Sequential(*layers)


def Upscale_Block(in_channels):
    """Upscaling using ConvTranspose"""
    layers = []
    # ConvTranspose layer
    layers.append(
        ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    )
    return nn.Sequential(*layers)


def OutConv_Block(in_channels, out_channels, bn_layer=True):
    layers = []
    # Convolution layer
    layers.append(Conv2d(in_channels, out_channels, kernel_size=1))
    if bn_layer:
        layers.append(BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
