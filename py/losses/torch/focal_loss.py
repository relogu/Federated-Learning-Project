import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        weight=None,
        size_average=True,
        ):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        inputs,
        targets,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        ):
        
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1-bce_exp)**gamma * bce
                       
        return focal_loss
