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
    ConvNext Decoders
"""


class CNDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(CNDecoder, self).__init__()
        self.conv_7_row = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernels[0],
                stride=strides[0],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernels[0],
                stride=strides[0],
            ),
        )

        self.conv_8_row = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=kernels[1], stride=strides[1]
        )

        self.upsample_1_row = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=kernels[1], stride=strides[1]
        )

        self.upsample_2_row = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[1],
            stride=strides[1],
        )

        self.upsample_3_row = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=256,
            kernel_size=kernels[3],
            stride=strides[3],
        )

        self.upsample_4_row = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=kernels[3], stride=strides[3]
        )

        self.upsample_5_row = nn.ConvTranspose2d(
            in_channels=128, out_channels=1, kernel_size=kernels[1], stride=strides[1]
        )

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_row(x)
        x = self.conv_8_row(x)

        out = self.upsample_1_row(x)
        out = torch.cat((out, pool_4_out), dim=1)
        out = self.upsample_2_row(out)
        out = torch.cat((out, pool_3_out), dim=1)
        out = self.upsample_3_row(out)
        out = self.upsample_4_row(out)
        out = self.upsample_5_row(out)

        return out
