import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearTied(nn.Module):
    def __init__(self, output_features, linear, bias=True):
        super(LinearTied, self).__init__()
        self.output_features = output_features
        self.linear = linear
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
            self._reset_bias()
        else:
            self.bias = None
    
    def _reset_bias(self):
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        reconstructed_input = F.linear(input, self.linear.weight.t(), self.bias)
        return reconstructed_input
