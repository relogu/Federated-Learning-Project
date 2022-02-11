import torch.nn as nn
import torch.nn.functional as F

class LovaszHingeLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        weight=None,
        size_average=True,
        ):
        super(LovaszHingeLoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(
        self, 
        inputs,
        targets,
        ):
        
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)
          
        lovasz = nn.lovasz_hinge(inputs, targets, per_image=False)                       
        return lovasz
