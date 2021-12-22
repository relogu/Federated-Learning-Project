import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TruncatedGaussianNoise(nn.Module):
    def __init__(self, shape, stddev: float = 0.05, rate: float = 0.5, unflatten: bool = False, seed=None, cuda: bool = False):
        super(TruncatedGaussianNoise, self).__init__()
        self.stddev = stddev
        self.rate = rate
        self.num_samples = int(self.rate*shape)
        self.seed = seed
        self.shape = shape
        self.unflatten = unflatten
        self.cuda = cuda
        if self.cuda:
            self.noise = Variable(torch.zeros(self.shape,self.shape).cuda())
        else:
            self.noise = Variable(torch.zeros(self.shape,self.shape))

    def forward(self, input):
        if self.unflatten:
            input = nn.Unflatten(1, (1, 28, 28))(input)
        if self.training:
            self.noise.data.normal_(0, std=self.stddev)
            if self.cuda:
                mask = (torch.rand(self.shape) > self.rate).cuda()
            else:
                mask = torch.rand(self.shape) > self.rate
            self.noise = self.noise.masked_fill(mask, 0)
            input = input + self.noise
        if self.unflatten:
            input = nn.Flatten()(input)
        return input
    