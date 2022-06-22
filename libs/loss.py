import torch
import torch.nn as nn
import config
from segmentation_models_pytorch import losses

class TDLoss(losses.DiceLoss):
    def __init__(self, loss = config.loss):
        super(TDLoss, self).__init__('binary')
        if loss == 'BCELoss':
            self.loss = nn.BCEWithLogitsLoss()

        elif loss == 'DiceLoss':
            self.loss = losses.DiceLoss('binary')

        elif loss == 'JaccardLoss':
            self.loss = losses.JaccardLoss('binary', smooth = 0.75)

        else:
            raise Exception('Invalid Loss Function')

    def forward(self, table_pred, table_gt,):
        table_loss = self.loss(table_pred, table_gt)        
        return table_loss