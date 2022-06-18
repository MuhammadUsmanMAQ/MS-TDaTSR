""" 
    Common library imports for training
        a model through pytorch 
"""
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision.models import resnet50, resnet101
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

"""
    Encoder definition
"""
class ConvNext(nn.Module):
    def __init__(self, scale = 'small', pretrained = True, requires_grad = True):
        super(ConvNext, self).__init__()

        if scale == 'tiny':
            convnext = convnext_tiny(pretrained = pretrained, progress = True)
        elif scale == 'small':
            convnext = convnext_small(pretrained = pretrained, progress = True)
        elif scale == 'base':
            convnext = convnext_base(pretrained = pretrained, progress = True)
        elif scale == 'large':
            convnext = convnext_large(pretrained = pretrained, progress = True)
        else:
            raise Exception('Invalid Scale for ConvNeXt')

        self.body = create_feature_extractor(
            convnext, return_nodes= {f'features.{k}': str(v)
                             for v, k in enumerate([5, 6, 7])}
        )

    def forward(self, x):
        out_1 = self.body(x)['0'] # torch.Size([1, 384, 64, 48])
        out_2 = self.body(x)['1'] # torch.Size([1, 768, 32, 24])
        out_3 = self.body(x)['2'] # torch.Size([1, 768, 32, 24])

        return out_1, out_2, out_3

class ResNet(torch.nn.Module):
    def __init__(self, arctype = '50', pretrained = True, requires_grad = True):
        super(ResNet, self).__init__()
        if arctype == '50':
            resnet = resnet50(pretrained = pretrained, progress = True)
        elif arctype == '101':
            resnet = resnet101(pretrained = pretrained, progress = True)
        else:
            raise Exception('Use either ResNet 50 or 101.')

        self.body = create_feature_extractor(
            resnet, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([2, 3, 4])})

        # Dry run to get number of channels for FPN
        inp = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]

        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        out_1 = x['0'] # torch.Size([1, 256, 128, 96])
        out_2 = x['1'] # torch.Size([1, 256, 64, 48])
        out_3 = x['2'] # torch.Size([1, 256, 32, 24])

        return out_1, out_2, out_3