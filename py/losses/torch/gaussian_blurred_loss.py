from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_kernel(size: int = 1,
                     sigma: float = 2.,
                     dim: int = 2,
                     channels: int = 1):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

class GaussianBlurLayer(nn.Module):

    def __init__(self, kernel_size: int, unflatten: bool = True, cuda: bool = False):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = _gaussian_kernel(size=kernel_size)
        if cuda:
            self.kernel = torch.cuda.FloatTensor(self.kernel)
        self.unflatten = unflatten

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

        padding = int((self.kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        x = torch.squeeze(F.conv2d(x, self.kernel))#, groups=3))
        if self.unflatten:
            x = nn.Flatten()(x)

        return x

class GaussianBlurredLoss(nn.Module):

    def __init__(self,
                 kernel_size: int,
                 alpha: float = 1.0,
                 loss_fn: Optional[torch.nn.Module] = nn.MSELoss,
                 unflatten: bool = True,
                 cuda: bool = False):
        super(GaussianBlurredLoss, self).__init__()
        self.alpha = alpha
        self.loss = loss_fn()
        self.grad_layer = GaussianBlurLayer(
            kernel_size=kernel_size,
            unflatten=unflatten,
            cuda=cuda
        )

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return  self.alpha * self.loss(output_grad, gt_grad) + (1 -  self.alpha) * self.loss(output, gt_img)
