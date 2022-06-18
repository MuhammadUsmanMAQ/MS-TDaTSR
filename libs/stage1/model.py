import torch
import torch.nn as nn
import torchvision
from encoder import ResNet, ConvNext
from decoder import RNDecoder, CNDecoder
from torchinfo import summary
import config

"""
    Model Definition
"""
class TDModel(nn.Module):
    def __init__(self, encoder = 'ConvNext-small', decoder = 'CNDecoder', use_pretrained_model = True, basemodel_requires_grad = True):
        super(TDModel, self).__init__()
        
        if (encoder == 'ConvNext-tiny' and decoder == 'CNDecoder'):
            self.kernels = [(1,1), (2,2), (2,2), (8,8)]
            self.strides = [(1,1), (2,2), (2,2), (8,8)]
            self.in_channels = 768
            
            self.base_model = ConvNext(scale = 'tiny', pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [768, 384]
            self.table_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)

        elif (encoder == 'ConvNext-small' and decoder == 'CNDecoder'):
            self.kernels = [(1,1), (2,2), (2,2), (8,8)]
            self.strides = [(1,1), (2,2), (2,2), (8,8)]
            self.in_channels = 768
            
            self.base_model = ConvNext(scale = 'small', pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [768, 384]
            self.table_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)

        elif (encoder == 'ConvNext-base' and decoder == 'CNDecoder'):
            self.kernels = [(1,1), (2,2), (2,2), (8,8)]
            self.strides = [(1,1), (2,2), (2,2), (8,8)]
            self.in_channels = 1024
            
            self.base_model = ConvNext(scale = 'base', pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [1024, 512]
            self.table_decoder = CNDecoder(self.pool_channels, self.kernels, self.strides)

        elif (encoder == 'resnet-50' and decoder == 'RNDecoder'):
            self.kernels = [(1,1), (2,2), (4,4), (8,8)]
            self.strides = [(1,1), (2,2), (4,4), (8,8)]
            self.in_channels = 256
            
            self.base_model = ResNet(arctype = '50', pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [256, 256]
            self.table_decoder = RNDecoder(self.pool_channels, self.kernels, self.strides)

        elif (encoder == 'resnet-101' and decoder == 'RNDecoder'):
            self.kernels = [(1,1), (2,2), (4,4), (8,8)]
            self.strides = [(1,1), (2,2), (4,4), (8,8)]
            self.in_channels = 256
            
            self.base_model = ResNet(arctype = '101', pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [256, 256]
            self.table_decoder = RNDecoder(self.pool_channels, self.kernels, self.strides)

        else:
            raise Exception('Invalid Enc-Dec Paramaters')

        # Common Layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7))

    def forward(self, x):
        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out)
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out)

        return table_out

if __name__ == '__main__':
    x = torch.randn(1, 3, 1024, 768)
    model = TDModel(encoder = 'resnet-50', decoder = 'RNDecoder', use_pretrained_model = True, basemodel_requires_grad = True)
    model(x)
    summary(model, input_size= (config.batch_size, 3, 1024, 768))