#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential, Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

from py.udec.net import DenseTied, ClusteringLayer


def create_autoencoder(dims, act='relu', init='glorot_uniform', verbose=False):
    """
    Fully connected auto-encoder model, symmetric, using DenseTied layers.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
        init: initializator of the layers' weights
        verbose: print some useful information on the layers' complexity
    return:
        (autoencoder, encoder, decoder), Model of autoencoder, Model of encoder, Model of decoder
    """
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    if verbose:
        print('List of encoder dimensions: {}'.format(encoder_dims))
        print('List of decoder dimensions: {}'.format(decoder_dims))
    # input data
    input_img = InputLayer(shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  #kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0),
                  kernel_initializer=init,
                  # kernel_constraint=UnitNorm(axis=0),
                  use_bias=False,  # True,
                  name='encoder_%d' % i)
        if verbose:
            print('Encoder Layer {}: {} with dims {}'.format(
                'encoder_%d' % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        if i == len(decoder_dims)-1:
            act = 'sigmoid'
        x = DenseTied(tied_to=encoder_layers[len(encoder_layers)-1-i],
                      units=decoder_dims[i],
                      activation=act,
                      #kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[len(encoder_dims)-1-i], weightage=1., axis=1),
                      # kernel_initializer=init,
                      use_bias=False,  # True,
                      # kernel_constraint=UnitNorm(axis=1),
                      name='decoder_%d' % i)
        if verbose:
            print('Decoder Layer {}: {} with dims {}, tied to {}'.format(
                'decoder_%d' % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print('Encoder Layers: {}'.format(encoder_layers))
        print('Decoder Layers: {}'.format(decoder_layers))
        print('Autoencoder Layers: {}'.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_prob_autoencoder(dims,
                            prob_layer=tfpl.IndependentBernoulli,
                            distr=tfd.Bernoulli.logits,
                            act=tf.nn.leaky_relu,
                            init='glorot_uniform',
                            verbose=False):
    """
    Fully connected auto-encoder model, symmetric, using DenseTied layers with a probabilistic layer on top of the decoder.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        prob_layer: probabilistic layer from tensorflow probability to put on top of the decoder
        distr: probability distribution onto which the probailistic layer is built
        act: activation, not applied to Input, Hidden and Output layers
        init: initializator of the layers' weights
        verbose: print some useful information on the layers' complexity
    return:
        (autoencoder, encoder, decoder), Model of autoencoder, Model of encoder, Model of decoder
    """
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    if verbose:
        print('List of encoder dimensions: {}'.format(encoder_dims))
        print('List of decoder dimensions: {}'.format(decoder_dims))
    # input data
    input_img = InputLayer(shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  #kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0),
                  kernel_initializer=init,
                  # kernel_constraint=UnitNorm(axis=0),
                  use_bias=False,  # True,
                  name='encoder_%d' % i)
        if verbose:
            print('Encoder Layer {}: {} with dims {}'.format(
                'encoder_%d' % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        if i == len(decoder_dims)-1:
            act = 'sigmoid'
        x = DenseTied(tied_to=encoder_layers[len(encoder_layers)-1-i],
                      units=decoder_dims[i],
                      activation=act,
                      #kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[len(encoder_dims)-1-i], weightage=1., axis=1),
                      # kernel_initializer=init,
                      use_bias=False,  # True,
                      # kernel_constraint=UnitNorm(axis=1),
                      name='decoder_%d' % i)
        if verbose:
            print('Decoder Layer {}: {} with dims {}, tied to {}'.format(
                'decoder_%d' % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)
    decoder_layers.append(
        prob_layer(dims[0], distr))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print('Encoder Layers: {}'.format(encoder_layers))
        print('Decoder Layers: {}'.format(decoder_layers))
        print('Autoencoder Layers: {}'.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_clustering_model(n_clusters, encoder):
    clustering_layer = ClusteringLayer(
        n_clusters, name='clustering')(encoder.output)
    return Model(inputs=encoder.input, outputs=clustering_layer)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
