#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:47:12 2021

@author: relogu
"""
import torch
import torch.nn as nn

from py.clustergan.util import initialize_weights, softmax
from py.clustergan.bsn import StochasticBinaryActivation

network_setup_string = "Setting up {}...\n"


class GeneratorDense(nn.Module):

    def __init__(self, latent_dim, n_c, gen_dims, x_shape, use_binary=False, verbose=False):
        super(GeneratorDense, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.gen_dims = gen_dims
        self.x_shape = x_shape
        self.verbose = verbose
        
        if use_binary:
            self.act = StochasticBinaryActivation
        else:
            self.act = nn.Sigmoid
        

        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim + self.n_c, self.gen_dims[0]),
            nn.BatchNorm1d(self.gen_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.gen_dims[0], self.gen_dims[1]),
            nn.BatchNorm1d(self.gen_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(self.gen_dims[1], self.gen_dims[2]),
            nn.BatchNorm1d(self.gen_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.gen_dims[2], self.gen_dims[3]),
            self.act(),
        )

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        #x_gen = x_gen.view(x_gen.size(0), self.x_shape)
        return x_gen


class EncoderDense(nn.Module):

    def __init__(self, latent_dim, enc_dims, n_c, verbose=False):
        super(EncoderDense, self).__init__()

        self.name = 'encoder'
        self.latent_dim = latent_dim
        self.enc_dims = enc_dims
        self.n_c = n_c
        self.verbose = verbose

        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.enc_dims[0], self.enc_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[1], self.enc_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[2], self.enc_dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[3], latent_dim + n_c)
        )

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # continuous components
        zn = z[:, 0:self.latent_dim]
        # one-hot components
        zc_logits = z[:, self.latent_dim:]
        # Softmax on one-hot component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class DiscriminatorDense(nn.Module):

    def __init__(self, disc_dims, wass_metric=False, verbose=False):
        super(DiscriminatorDense, self).__init__()

        self.name = 'discriminator'
        self.disc_dims = disc_dims
        self.wass = wass_metric
        self.verbose = verbose

        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.disc_dims[0], self.disc_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[1], self.disc_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[2], self.disc_dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[3], 1)
        )

        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity
