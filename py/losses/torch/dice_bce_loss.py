import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        weight=None,
        size_average=True,
        ):
        super(DiceBCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        inputs,
        targets,
        smooth: float = 1,
        ):
        
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = bce + dice_loss
        
        return dice_bce
