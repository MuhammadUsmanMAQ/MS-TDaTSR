import torch
import torch.nn as nn

class TDLoss(nn.Module):
    def __init__(self):
        super(TDLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, table_pred, table_gt,):
        table_loss = self.bce(table_pred, table_gt)        
        return table_loss