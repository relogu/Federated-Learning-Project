from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLayer(nn.Module):

    def __init__(self, factor: float = 2.0, unflatten: bool = True, return_mask: bool = True, cuda: bool = False):
        super(SobelLayer, self).__init__()
        self.unflatten = unflatten
        self.return_mask = return_mask
        self.cuda = cuda

        if self.cuda:
            Gx = torch.cuda.FloatTensor([[factor, 0.0, -factor], [2*factor, 0.0, -2*factor], [factor, 0.0, -factor]]).unsqueeze(0).unsqueeze(0)
            Gy = torch.cuda.FloatTensor([[factor, 2*factor, factor], [0.0, 0.0, 0.0], [-factor, -2*factor, -factor]]).unsqueeze(0).unsqueeze(0)
            Gxy = torch.cuda.FloatTensor([[2*factor, factor, 0.0], [factor, 0.0, -factor], [0.0, -factor, -2*factor]]).unsqueeze(0).unsqueeze(0)
            Gyx = torch.cuda.FloatTensor([[0.0, factor, 2*factor], [-factor, 0.0, factor], [-2*factor, -factor, 0.0]]).unsqueeze(0).unsqueeze(0)
        else:
            Gx = torch.FloatTensor([[factor, 0.0, -factor], [2*factor, 0.0, -2*factor], [factor, 0.0, -factor]]).unsqueeze(0).unsqueeze(0)
            Gy = torch.FloatTensor([[factor, 2*factor, factor], [0.0, 0.0, 0.0], [-factor, -2*factor, -factor]]).unsqueeze(0).unsqueeze(0)
            Gxy = torch.FloatTensor([[2*factor, factor, 0.0], [factor, 0.0, -factor], [0.0, -factor, -2*factor]]).unsqueeze(0).unsqueeze(0)
            Gyx = torch.FloatTensor([[0.0, factor, 2*factor], [-factor, 0.0, factor], [-2*factor, -factor, 0.0]]).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=Gx, requires_grad=False)
        self.weight_y = nn.Parameter(data=Gy, requires_grad=False)
        self.weight_xy = nn.Parameter(data=Gxy, requires_grad=False)
        self.weight_yx = nn.Parameter(data=Gyx, requires_grad=False)
        if self.cuda:
            self.weight_x.cuda(non_blocking=True)
            self.weight_y.cuda(non_blocking=True)
            self.weight_xy.cuda(non_blocking=True)
            self.weight_yx.cuda(non_blocking=True)
            

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, input):
        
        if self.unflatten:
            input = nn.Unflatten(1, (1, 28, 28))(input)
        if input.shape[1] == 3:
            input = self.get_gray(input)
        
        x = F.conv2d(input, self.weight_x, padding=1)
        y = F.conv2d(input, self.weight_y, padding=1)
        xy = F.conv2d(input, self.weight_xy, padding=1)
        yx = F.conv2d(input, self.weight_yx, padding=1)
        output = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(xy, 2) + torch.pow(yx, 2) + 1e-6)
        
        if self.unflatten:
            output = nn.Flatten()(output)
        
        batch_norm = nn.BatchNorm1d(output.shape[1], track_running_stats=False)
        if self.cuda:
            batch_norm.cuda()
        output = batch_norm(output)
        #output = F.batch_norm(output)
        mask = (output > 1e-3)
        if self.return_mask:
            return output, mask
        else:
            return output
    
class Sobel(nn.Module):
    def __init__(self, factor: float = 2.0):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[factor, 0.0, -factor], [2*factor, 0.0, -2*factor], [factor, 0.0, -factor]])
        Gy = torch.tensor([[factor, 2*factor, factor], [0.0, 0.0, 0.0], [-factor, -2*factor, -factor]])
        Gxy = torch.tensor([[2*factor, factor, 0.0], [factor, 0.0, -factor], [0.0, -factor, -2*factor]])
        Gyx = torch.tensor([[0.0, factor, 2*factor], [-factor, 0.0, factor], [-2*factor, -factor, 0.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0), Gxy.unsqueeze(0), Gyx.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x = nn.Flatten()(x)
        print(img.shape)
        print(x.shape)
        x = x.view(img.size(0), img.size(1))
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x = x.view(img.size(0), img.size(1))
        return x

class SobelLoss(nn.Module):

    def __init__(self,
                 alpha: float = 1.0,
                 loss_fn: Optional[torch.nn.Module] = nn.L1Loss,
                 apply_sigmoid: bool = False,
                 unflatten: bool = True,
                 return_mask: bool = True,
                 cuda: bool = True):
        super(SobelLoss, self).__init__()
        self.alpha = alpha
        self.loss = loss_fn()
        self.apply_sigmoid = apply_sigmoid
        self.grad_layer = SobelLayer(unflatten=unflatten, return_mask=return_mask, cuda=cuda)
        self.loss.reduction = 'none'

    def forward(self, output, gt_img):
        if self.apply_sigmoid:
            output = nn.Sigmoid()(output)
        output_grad, _ = self.grad_layer(output)
        gt_grad, mask = self.grad_layer(gt_img)
        filt_loss = torch.sum(mask * self.loss(output_grad, gt_grad)) / torch.sum(mask)
        unfilt_loss = torch.sum(mask * self.loss(output, gt_img)) / torch.sum(mask)
        return  self.alpha * filt_loss + (1 -  self.alpha) * unfilt_loss
