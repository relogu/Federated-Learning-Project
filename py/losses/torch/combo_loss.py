import torch
import torch.nn as nn

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 # weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        ):
        super(ComboLoss, self).__init__()

    def forward(
        self,
        inputs,
        targets,
        smooth: float = 1,
        alpha: float = ALPHA,
        ce_ratio: float = CE_RATIO,
        eps: float = 1e-9,
        ):
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)
        
        return combo
