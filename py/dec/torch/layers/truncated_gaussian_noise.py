import torch
import torch.nn as nn
from torch.autograd import Variable

class TruncatedGaussianNoise(nn.Module):
    def __init__(self,
                 shape,
                 stddev: float = 0.05,
                 rate: float = 0.5,
                 seed=None,
                 device: str = 'cpu',
                 ):
        super(TruncatedGaussianNoise, self).__init__()
        self.stddev = stddev
        self.rate = rate
        self.num_samples = int(self.rate*shape)
        self.seed = seed
        self.shape = shape
        self.device = device
        
        self.noise = Variable(torch.zeros(self.shape)).to(self.device)

    def forward(self, input):
        if self.training:
            self.noise.data.normal_(0, std=self.stddev)
            
            mask = (torch.rand(self.shape) > self.rate).to(self.device)
            self.noise = self.noise.masked_fill(mask, 0)
            input = input + self.noise
            
            mask = (input > 1.0).to(self.device)
            input = input.masked_fill(mask, 1.0)
            
            mask = (input < 0.0).to(self.device)
            input = input.masked_fill(mask, 0.0)
        return input
    