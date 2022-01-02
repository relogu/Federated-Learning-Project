from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianLayer(nn.Module):

    def __init__(self, unflatten: bool = True, return_mask: bool = True, cuda: bool = False):
        super(LaplacianLayer, self).__init__()
        self.unflatten = unflatten
        self.return_mask = return_mask 
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        if cuda:
            kernel = torch.cuda.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        else:
            kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        if cuda:
            self.weight.cuda(non_blocking=True)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        
        if self.unflatten:
            x = nn.Unflatten(1, (1, 28, 28))(x)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x = F.conv2d(x, self.weight, padding=1)
        if self.unflatten:
            x = nn.Flatten()(x)
        mask = (x > 0)
        if self.return_mask:
            return x, mask
        else:
            return x
    
class Laplacian(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

        G = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]]
            ).unsqueeze(0).unsqueeze(0)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class LaplacianLoss(nn.Module):

    def __init__(self,
                 alpha: float = 1.0,
                 loss_fn: Optional[torch.nn.Module] = nn.L1Loss,
                 apply_sigmoid: bool = False,
                 unflatten: bool = True,
                 return_mask: bool = True,
                 cuda: bool = True):
        super(LaplacianLoss, self).__init__()
        self.alpha = alpha
        self.loss = loss_fn()
        self.apply_sigmoid = apply_sigmoid
        self.grad_layer = LaplacianLayer(unflatten=unflatten, return_mask=return_mask, cuda=cuda)
        self.loss.reduction = 'none'

    def forward(self, output, gt_img):
        if self.apply_sigmoid:
            output = nn.Sigmoid()(output)
        output_grad, _ = self.grad_layer(output)
        gt_grad, mask = self.grad_layer(gt_img)
        filt_loss = torch.mean(mask * self.loss(output_grad, gt_grad))
        unfilt_loss = torch.mean(mask * self.loss(output, gt_img))
        return  self.alpha * filt_loss + (1 -  self.alpha) * unfilt_loss
