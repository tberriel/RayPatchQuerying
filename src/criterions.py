import torch.nn as nn

class AbsoluteRelativeDifference(nn.Module):
    def __init__(self, pred_log = False, target_log = False):
        super().__init__()
        self.pred_log = pred_log
        self.target_log = target_log
    def forward(self, pred, target):
        if self.pred_log:
            pred = pred.exp()
        if self.target_log:
            target = target.exp()

        return ((pred-target).abs()/target).mean()
  
class SquareRelativeDifference(nn.Module):
    def __init__(self, pred_log = False, target_log = False):
        super().__init__()
        self.pred_log = pred_log
        self.target_log = target_log
    
    def forward(self, pred, target):  
        if self.pred_log:
            pred = pred.exp()
        if self.target_log:
            target = target.exp()
                  
        return ((pred-target).square()/target).mean()
            
