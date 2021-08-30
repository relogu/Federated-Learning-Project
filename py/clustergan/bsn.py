#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:47:12 2021

@author: relogu
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from py.distributions.bernoulli import Bernoulli as BernoulliREINFORCE
from py.distributions.round import Round as RoundREINFORCE


class BinaryNet(nn.Module):

    def __init__(self, input_dim, output_dim, mode='Deterministic', estimator='ST'):
        super(BinaryNet, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']

        self.mode = mode
        self.estimator = estimator

        if self.mode == 'Deterministic':
            self.act = DeterministicBinaryActivation(estimator=estimator)
        elif self.mode == 'Stochastic':
            self.act = StochasticBinaryActivation(estimator=estimator)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        #x, slope = input
        #x_act = self.act((x, slope))
        x_act = self.act(input)
        x_fc = self.fc(x_act)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out


class DeterministicBinaryActivation(nn.Module):

    def __init__(self, estimator='ST'):
        super(DeterministicBinaryActivation, self).__init__()

        assert estimator in ['ST', 'REINFORCE']

        self.estimator = estimator
        self.act = Hardsigmoid()

        if self.estimator == 'ST':
            self.binarizer = RoundST
        elif self.estimator == 'REINFORCE':
            self.binarizer = RoundREINFORCE

    def forward(self, input):
        #x, slope = input
        #probs = self.act(slope * x)
        probs = self.act(input)
        out = self.binarizer(probs)
        if self.estimator == 'REINFORCE':
            out = out.sample()
        return out


class StochasticBinaryActivation(nn.Module):

    def __init__(self, estimator='ST'):
        super(StochasticBinaryActivation, self).__init__()

        assert estimator in ['ST', 'REINFORCE']

        self.estimator = estimator
        self.act = Hardsigmoid()

        if self.estimator == 'ST':
            self.binarizer = BernoulliST
        elif self.estimator == 'REINFORCE':
            self.binarizer = BernoulliREINFORCE

    def forward(self, input):
        #x, slope = input
        #probs = self.act(slope * x)
        probs = self.act(input)
        out = self.binarizer(probs)
        if self.estimator == 'REINFORCE':
            out = out.sample()
        return out


class Hardsigmoid(nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = nn.Hardtanh()

    def forward(self, x):
        return (self.act(x) + 1.0) / 2.0


class RoundFunctionST(Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):
        """Forward pass
        Parameters
        ==========
        :param input: input tensor
        Returns
        =======
        :return: a tensor which is round(input)"""

        # We can cache arbitrary Tensors for use in the backward pass using the
        # save_for_backward method.
        # ctx.save_for_backward(input)

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass we receive a tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of the
        loss with respect to the input.
        Parameters
        ==========
        :param grad_output: tensor that stores the gradients of the loss wrt. output
        Returns
        =======
        :return: tensor that stores the gradients of the loss wrt. input"""

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # input, weight, bias = ctx.saved_variables

        return grad_output


class BernoulliFunctionST(Function):

    @staticmethod
    def forward(ctx, input):

        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
    
        return grad_output


RoundST = RoundFunctionST.apply
BernoulliST = BernoulliFunctionST.apply
