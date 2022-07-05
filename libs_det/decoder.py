""" 
    Common library imports for training
        a model through pytorch 
"""
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
import config

"""
    ConvNext Decoder
"""

class CNDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(CNDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=kernels[0],
            stride=strides[0],
        )

        self.conv_8_table = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=256,
            kernel_size=kernels[3],
            stride=strides[3],
        )

        self.upsample_4_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[3],
            stride=strides[3],
        )

        self.upsample_5_table = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=1,
            kernel_size=kernels[1],
            stride=strides[1],
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)
        x = self.conv_8_table(x)

        out = self.upsample_1_table(x)
        out = torch.cat((out, pool_4_out), dim=1)
        out = self.upsample_2_table(out)
        out = torch.cat((out, pool_3_out), dim=1)
        out = self.upsample_3_table(out)
        out = self.upsample_4_table(out)
        out = self.upsample_5_table(out)

        return out


"""
    ResNet Decoder
"""

class RNDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(RNDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[0],
            stride=strides[0],
        )

        self.conv_8_table = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_1_1_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_1_2_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[2],
            stride=strides[2],
        )

        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3],
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)
        x = self.conv_8_table(x)

        out = self.upsample_1_1_table(x)
        out = self.upsample_1_2_table(x)
        out = torch.cat((out, pool_4_out), dim=1)
        out = self.upsample_2_table(out)
        out = torch.cat((out, pool_3_out), dim=1)
        out = self.upsample_3_table(out)

        return out


"""
    EfficientNet Decoder
"""

class ENDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ENDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=kernels[0],
            stride=strides[0],
        )

        self.conv_8_table = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=512 + channels[0],
            out_channels=1024,
            kernel_size=kernels[0],
            stride=strides[0],
        )

        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=1024 + channels[1],
            out_channels=512,
            kernel_size=kernels[2],
            stride=strides[2],
        )

        self.upsample_4_table = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=512,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_5_table = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2],
        )

        self.upsample_6_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=1,
            kernel_size=kernels[1],
            stride=strides[1],
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)
        x = self.conv_8_table(x)

        out = self.upsample_1_table(x)
        out = torch.cat((out, pool_4_out), dim=1)
        out = self.upsample_2_table(out)
        out = torch.cat((out, pool_3_out), dim=1)
        out = self.upsample_3_table(out)
        out = self.upsample_4_table(out)
        out = self.upsample_5_table(out)
        out = self.upsample_6_table(out)

        return out
