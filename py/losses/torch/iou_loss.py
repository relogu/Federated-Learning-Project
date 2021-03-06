import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        weight=None,
        size_average=True,
        ):
        super(IoULoss, self).__init__()
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
        
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
                
        return 1 - iou
