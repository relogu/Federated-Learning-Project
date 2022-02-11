import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        ):
        super(FocalTverskyLoss, self).__init__()

    def forward(
        self,
        inputs,
        targets,
        smooth: float = 1,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        ):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()    
        fp = ((1-targets) * inputs).sum()
        fn = (targets * (1-inputs)).sum()
        
        tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)  
        focal_tversky = (1 - tversky)**gamma
                       
        return focal_tversky
