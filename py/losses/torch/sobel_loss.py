from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLayer(nn.Module):

    def __init__(self, unflatten: bool = True, cuda: bool = False):
        super(SobelLayer, self).__init__()
        self.unflatten = unflatten 
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        if cuda:
            kernel_h = torch.cuda.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
            kernel_v = torch.cuda.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        else:
            kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
            kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        if cuda:
            self.weight_h.cuda(non_blocking=True)
            self.weight_v.cuda(non_blocking=True)

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

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        if self.unflatten:
            x = nn.Flatten()(x)

        return x

class SobelLoss(nn.Module):

    def __init__(self,
                 loss_fn: Optional[torch.nn.Module] = nn.MSELoss,
                 unflatten: bool = True,
                 cuda: bool = True):
        super(SobelLoss, self).__init__()
        self.loss = loss_fn()
        self.grad_layer = SobelLayer(unflatten=unflatten, cuda=cuda)

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)
