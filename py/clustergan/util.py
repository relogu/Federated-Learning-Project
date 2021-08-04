#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:47:12 2021

@author: relogu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# Sample a random latent space vector


def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False, cuda=False):

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c)
            ), "Requested class %i outside bounds." % fix_class
    TENSOR = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Sample noise as generator input, zn
    zn = Variable(TENSOR(0.75*np.random.normal(0, 1,
                  (shape, latent_dim))), requires_grad=req_grad)

    # zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_ft = TENSOR(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda() if cuda else zc_idx.random_(n_c)
        zc_ft = zc_ft.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_ft[:, fix_class] = 1

        if cuda:
            zc_idx = zc_idx.cuda()
            zc_ft = zc_ft.cuda()

    zc = Variable(zc_ft, requires_grad=req_grad)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(net_d, real_data, generated_data, cuda=False):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.cuda()

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = net_d(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(
                           ) if cuda else torch.ones(prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.ConvTranspose2d) \
                or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)
