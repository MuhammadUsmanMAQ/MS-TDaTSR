import torch
import torch.nn as nn
import torchvision
from encoder import ConvNext
from decoder import CNDecoder
from torchinfo import summary
import config

"""
    Model Definition
"""


class TDModel(nn.Module):
    def __init__(
        self,
        encoder=config.encoder,
        decoder=config.decoder,
        use_pretrained_model=True,
        basemodel_requires_grad=True,
    ):
        super(TDModel, self).__init__()

        # ConvNeXt
        if encoder == "ConvNext-tiny" and decoder == "CNDecoder":
            self.kernels = [(1, 1), (2, 2), (16, 16), (4, 4)]
            self.strides = [(1, 1), (2, 2), (16, 16), (4, 4)]
            self.in_channels = 768

            self.base_model = ConvNext(
                scale="tiny",
                pretrained=use_pretrained_model,
                requires_grad=basemodel_requires_grad,
            )
            self.pool_channels = [768, 384]

            self.row_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.col_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.rowheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.colheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.spanning_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )

        elif encoder == "ConvNext-small" and decoder == "CNDecoder":
            self.kernels = [(1, 1), (2, 2), (16, 16), (4, 4)]
            self.strides = [(1, 1), (2, 2), (16, 16), (4, 4)]
            self.in_channels = 768

            self.base_model = ConvNext(
                scale="small",
                pretrained=use_pretrained_model,
                requires_grad=basemodel_requires_grad,
            )
            self.pool_channels = [768, 384]

            self.row_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.col_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.rowheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.colheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.spanning_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )

        elif encoder == "ConvNext-base" and decoder == "CNDecoder":
            self.kernels = [(1, 1), (2, 2), (2, 2), (8, 8)]
            self.strides = [(1, 1), (2, 2), (2, 2), (8, 8)]
            self.in_channels = 1024

            self.base_model = ConvNext(
                scale="base",
                pretrained=use_pretrained_model,
                requires_grad=basemodel_requires_grad,
            )
            self.pool_channels = [1024, 512]

            self.row_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.col_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)
            self.rowheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.colheader_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )
            self.spanning_decoder = CNDecoder(
                self.pool_channels, self.kernels, self.strides
            )

        # Exception
        else:
            raise Exception("Invalid Enc-Dec Paramaters")

        # Common Layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=512, kernel_size=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
        )

    def forward(self, x):
        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out)
        tr_out = self.row_decoder(conv_out, pool_3_out, pool_4_out)
        tc_out = self.col_decoder(conv_out, pool_3_out, pool_4_out)
        trh_out = self.rowheader_decoder(conv_out, pool_3_out, pool_4_out)
        tch_out = self.colheader_decoder(conv_out, pool_3_out, pool_4_out)
        ts_out = self.spanning_decoder(conv_out, pool_3_out, pool_4_out)

        return tr_out, tc_out, trh_out, tch_out, ts_out


if __name__ == "__main__":
    model = TDModel(use_pretrained_model=True, basemodel_requires_grad=True)
    summary(model, input_size=(1, 3, 384, 512))
