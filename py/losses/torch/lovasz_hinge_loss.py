import torch.nn as nn
import torch.nn.functional as F

class LovaszHingeLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        ):
        super(LovaszHingeLoss, self).__init__()

    def forward(
        self, 
        inputs,
        targets,
        ):
        inputs = F.sigmoid(inputs)    
        lovasz = nn.lovasz_hinge(inputs, targets, per_image=False)                       
        return lovasz
