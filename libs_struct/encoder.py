""" 
    Common library imports for training
        a model through pytorch 
"""
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
)
from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)
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
    def __init__(self, scale="small", pretrained=True, requires_grad=True):
        super(ConvNext, self).__init__()

        if scale == "tiny":
            convnext = convnext_tiny(pretrained=pretrained, progress=True)
        elif scale == "small":
            convnext = convnext_small(pretrained=pretrained, progress=True)
        elif scale == "base":
            convnext = convnext_base(pretrained=pretrained, progress=True)
        elif scale == "large":
            convnext = convnext_large(pretrained=pretrained, progress=True)
        else:
            raise Exception("Invalid Scale for ConvNeXt")

        self.body = create_feature_extractor(
            convnext,
            return_nodes={f"features.{k}": str(v) for v, k in enumerate([5, 6, 7])},
        )
        self.maxpool = nn.MaxPool2d(1, 2, 0)

    def forward(self, x):
        x = self.body(x)
        out_1 = self.maxpool(x["0"])  # torch.Size([1, 384, 64, 48])
        out_2 = self.maxpool(x["1"])  # torch.Size([1, 768, 32, 24])
        out_3 = self.maxpool(x["2"])  # torch.Size([1, 768, 32, 24])

        return out_1, out_2, out_3
