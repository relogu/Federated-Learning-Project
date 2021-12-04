#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
from typing import List, Dict
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.constraints import UnitNorm
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

from py.dec.constraints.uncoll_feat import UncorrelatedFeaturesConstraint

from .layers import DenseTied, ClusteringLayer, FlippingNoise, TruncatedGaussianNoise
from .constraints import WeightsOrthogonalityConstraint

# some string for verbose outputs
encoder_layer_name = 'encoder_%d'
decoder_layer_name = 'decoder_%d'
list_enc_dim = 'List of encoder dimensions: {}'
list_dec_dim = 'List of decoder dimensions: {}'
enc_verb = 'Encoder Layer {}: {} with dims {}'
dec_verb = 'Decoder Layer {}: {} with dims {}'
dec_tied_verb = 'Decoder Layer {}: {} with dims {}, tied to {}'
enc_layers = 'Encoder Layers: {}'
dec_layers = 'Decoder Layers: {}'
ae_layers = 'Autoencoder Layers: {}'


def create_denoising_autoencoder(dims,
                                 up_freq=None,
                                 act='relu',
                                 init='glorot_uniform',
                                 noise_rate=0.1,
                                 dropout_rate=0.0,
                                 bias=False,
                                 verbose=False):
    """
    Fully connected auto-encoder model, symmetric, using DenseTied layers.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        up_freq: list of frequencies of the up values in the training set
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
        print(list_enc_dim.format(encoder_dims))
        print(list_dec_dim.format(decoder_dims))
    # input data
    input_img = InputLayer(input_shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(input_shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        if i == len(encoder_dims)-1:
            act = None
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  kernel_initializer=init,
                  use_bias=bias,
                  name=encoder_layer_name % i)
        if verbose:
            print(enc_verb.format(
                encoder_layer_name % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        if i == len(decoder_dims)-1:
            act = 'sigmoid'
        x = Dense(units=decoder_dims[i],
                  activation=act,
                  kernel_initializer=init,
                  use_bias=bias,
                  name=decoder_layer_name % i)
        if verbose:
            print(dec_verb.format(
                decoder_layer_name % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)

    # adding flipping noise
    if noise_rate > 0.0:
        encoder_layers.insert(1, FlippingNoise(
            up_frequencies=up_freq, rate=noise_rate))

    # adding dropout
    if dropout_rate > 0.0:
        if verbose:
            print('Adding dropout of rate {}'.format(dropout_rate))
        idx = np.arange(start=3, stop=int((2*len(dims))-2), step=2)
        for i in idx:
            encoder_layers.insert(i, Dropout(rate=dropout_rate))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print(enc_layers.format(encoder_layers))
        print(dec_layers.format(decoder_layers))
        print(ae_layers.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_tied_denoising_autoencoder(dims,
                                      up_freq: List[float] = None,
                                      b_idx: List[int] = None,
                                      act='relu',
                                      init='glorot_uniform',
                                      noise_rate=0.1,
                                      dropout_rate=0.0,
                                      use_bias=False,
                                      ortho=False,
                                      u_norm=False,
                                      verbose=False):
    """
    Fully connected denoising stacked auto-encoder model, symmetric, using DenseTied layers.
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
        print(list_enc_dim.format(encoder_dims))
        print(list_dec_dim.format(decoder_dims))
    # input data
    input_img = InputLayer(input_shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(input_shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        # k_reg = WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0) if ortho else None
        k_reg, k_con = None, None
        k_con = UnitNorm(axis=0) if u_norm else None
        activation = act
        if i == len(encoder_dims)-1:
            k_reg = WeightsOrthogonalityConstraint(
                encoder_dims[i], weightage=1., axis=0) if ortho else None
            activation = None
        x = Dense(units=encoder_dims[i],
                  activation=activation,
                  kernel_regularizer=k_reg,
                  kernel_initializer=init,
                  kernel_constraint=k_con,
                  use_bias=use_bias,
                  name=encoder_layer_name % i)
        if verbose:
            print(enc_verb.format(
                encoder_layer_name % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        #k_reg = WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0) if ortho else None
        activation = act
        if i == len(encoder_dims)-1:
            act = 'sigmoid'
        x = DenseTied(tied_to=encoder_layers[len(encoder_layers)-1-i],
                      units=decoder_dims[i],
                      activation=activation,
                      #kernel_regularizer=k_reg,
                      #kernel_constraint=k_con,
                      use_bias=use_bias,
                      name=decoder_layer_name % i)
        if verbose:
            print(dec_tied_verb.format(
                decoder_layer_name % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)

    # adding flipping noise
    if noise_rate > 0.0:
        encoder_layers.insert(1, FlippingNoise(
            up_frequencies=up_freq,
            b_idx=b_idx,
            rate=noise_rate))
        # encoder_layers.insert(1, TruncatedGaussianNoise(
        #     stddev=0.1,
        #     rate=noise_rate,
        # ))

    # adding dropout
    if dropout_rate > 0.0:
        if verbose:
            print('Adding dropout of rate {}'.format(dropout_rate))
        idx = np.arange(start=3, stop=int((2*len(dims))-2), step=2)
        for i in idx:
            encoder_layers.insert(i, Dropout(rate=dropout_rate))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print(enc_layers.format(encoder_layers))
        print(dec_layers.format(decoder_layers))
        print(ae_layers.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_tied_prob_autoencoder(dims,
                                 prob_layer=tfpl.IndependentBernoulli,
                                 distr=tfd.Bernoulli.logits,
                                 act=tf.nn.leaky_relu,
                                 init='glorot_uniform',
                                 dropout=False,
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
        print(list_enc_dim.format(encoder_dims))
        print(list_dec_dim.format(decoder_dims))
    # input data
    input_img = InputLayer(input_shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(input_shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        activation = act
        if i == len(encoder_dims)-1:
            activation = None
        x = Dense(units=encoder_dims[i],
                  activation=activation,
                  # kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0),
                  kernel_initializer=init,
                  # kernel_constraint=UnitNorm(axis=0),
                  # use_bias=True,  # default False
                  name=encoder_layer_name % i)
        if verbose:
            print(enc_verb.format(
                encoder_layer_name % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        activation = act
        if i == len(decoder_dims)-1:
            activation = 'sigmoid'
        x = DenseTied(tied_to=encoder_layers[len(encoder_layers)-1-i],
                      units=decoder_dims[i],
                      activation=activation,
                      # kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[len(encoder_dims)-1-i], weightage=1., axis=1),
                      # kernel_initializer=init,
                      # use_bias=True,  # default False
                      # kernel_constraint=UnitNorm(axis=1),
                      name=decoder_layer_name % i)
        if verbose:
            print(dec_tied_verb.format(
                decoder_layer_name % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)
    decoder_layers.append(
        prob_layer(dims[0], distr))

    # adding dropout
    if dropout:
        encoder_layers.insert(1, Dropout(rate=0.2))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print(enc_layers.format(encoder_layers))
        print(dec_layers.format(decoder_layers))
        print(ae_layers.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_prob_autoencoder(dims,
                            prob_layer=tfpl.IndependentBernoulli,
                            distr=tfd.Bernoulli.logits,
                            act=tf.nn.leaky_relu,
                            init='glorot_uniform',
                            dropout=False,
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
        print(list_enc_dim.format(encoder_dims))
        print(list_dec_dim.format(decoder_dims))
    # input data
    input_img = InputLayer(input_shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(input_shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  kernel_initializer=init,
                  use_bias=True,
                  name=encoder_layer_name % i)
        if verbose:
            print(enc_verb.format(
                encoder_layer_name % i, x, encoder_dims[i]))
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        if i == len(decoder_dims)-1:
            act = 'sigmoid'
        x = Dense(units=decoder_dims[i],
                  activation=act,
                  kernel_initializer=init,
                  use_bias=True,
                  name=decoder_layer_name % i)
        if verbose:
            print(dec_tied_verb.format(
                decoder_layer_name % i,
                x,
                decoder_dims[i],
                encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)
    decoder_layers.append(
        prob_layer(dims[0], distr))

    # adding dropout
    if dropout:
        encoder_layers.insert(1, Dropout(rate=0.2))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    if verbose:
        print(enc_layers.format(encoder_layers))
        print(dec_layers.format(decoder_layers))
        print(ae_layers.format(autoencoder_layers))
        autoencoder.summary()
        encoder.summary()
        decoder.summary()

    return (autoencoder, encoder, decoder)


def create_dec_sae(dims,
                   unit_norm: bool = False,
                   w_ortho: bool = False,
                   uncorr_feat: bool = False,
                   activation='relu',
                   use_bias: bool = False,
                   noise_std: float = 0.1,
                   init='glorot_uniform',
                   dropout_rate: float = 0.2):
    """
    Fully connected auto-encoder model, symmetric, using DenseTied layers.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        up_freq: list of frequencies of the up values in the training set
        act: activation, not applied to Input, Hidden and Output layers
        init: initializator of the layers' weights
        verbose: print some useful information on the layers' complexity
    return:
        (autoencoder, encoder, decoder), Model of autoencoder, Model of encoder, Model of decoder
    """
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    # input data
    input_img = InputLayer(input_shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = InputLayer(input_shape=(dims[-1],), name='input_lbl')
    # encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        act = 'relu'
        if i == len(encoder_dims)-1:
            act = None
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  kernel_initializer=init,
                  use_bias=True,
                  name=encoder_layer_name % i)
        encoder_layers.append(x)

    # decoder
    decoder_layers = []
    decoder_layers.append(input_lbl)
    # internal layers in decoder
    for i in range(len(decoder_dims)):
        act = 'relu'
        if i == len(decoder_dims)-1:
            act = None
        x = DenseTied(tied_to=encoder_layers[len(encoder_layers)-1-i],
                      units=decoder_dims[i],
                      activation=act,
                      kernel_initializer=init,
                      use_bias=True,
                      name=decoder_layer_name % i)
        decoder_layers.append(x)

    # adding dropout
    if dropout_rate > 0.0:
        idx = np.arange(start=1, stop=int((2*len(dims))-2), step=2)
        for i in idx:
            encoder_layers.insert(i, Dropout(rate=dropout_rate))

    # adding gaussian noise
    if noise_std > 0.0:
        encoder_layers.insert(1, GaussianNoise(stddev=noise_std))

    # autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    return (autoencoder, encoder, decoder)


def create_autoencoder(net_arch, up_frequencies):
    if net_arch['binary']:
        if net_arch['tied']:
            return create_tied_prob_autoencoder(
                net_arch['dims'],
                init=net_arch['init'],
                dropout=net_arch['dropout'],
                act=net_arch['act'])
        else:
            return create_prob_autoencoder(
                net_arch['dims'],
                init=net_arch['init'],
                dropout=net_arch['dropout'],
                act=net_arch['act'])
    else:
        if net_arch['tied']:
            return create_tied_denoising_autoencoder(
                net_arch['dims'],
                up_freq=up_frequencies,
                b_idx=net_arch['b_idx'],
                init=net_arch['init'],
                dropout_rate=net_arch['dropout'],
                act=net_arch['act'],
                ortho=net_arch['ortho'],
                u_norm=net_arch['u_norm'],
                noise_rate=net_arch['ran_flip'],
                use_bias=net_arch['use_bias'])
        else:
            return create_denoising_autoencoder(
                net_arch['dims'],
                up_freq=up_frequencies,
                init=net_arch['init'],
                dropout_rate=net_arch['dropout'],
                act=net_arch['act'],
                noise_rate=net_arch['ran_flip'])


def create_denoising_autoencoder(
    flavor: str = 'real',
    dims: List[int] = None,
    activation='relu',
    w_init='glorot_uniform',
    is_tied: bool = True,
    u_norm_reg: bool = False,
    ortho_w_con: bool = False,
    uncoll_feat_reg: bool = False,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    noise_rate: float = 0.0,
    ran_flip_conf: Dict = None,
):
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    # input data
    input_data = InputLayer(input_shape=(dims[0],), name='input_data')
    # input features
    input_feat = InputLayer(input_shape=(dims[-1],), name='input_feat')

    # build encoder layers
    # encoder_layers = []
    # encoder_layers.append(input_data)
    encoder_layers = [input_data] + get_encoder_layers(
        dims=encoder_dims,
        activation=activation,
        w_init=w_init,
        u_norm_reg=u_norm_reg,
        ortho_w_con=ortho_w_con,
        uncoll_feat_reg=uncoll_feat_reg,
        use_bias=use_bias
    )
    # build decoder layers
    # decoder_layers = []
    # decoder_layers.append(input_feat)
    decoder_layers = [input_feat] + get_decoder_layers(
        dims=decoder_dims,
        activation=activation,
        is_tied=is_tied,
        w_init=w_init,
        u_norm_reg=u_norm_reg,
        ortho_w_con=ortho_w_con,
        use_bias=use_bias,
        output_activation=get_output_act(flavor),
        encoder_layers=encoder_layers
    )
    # noising layers
    encoder_layers = noise_layers_fn(
        layers=encoder_layers,
        flavor=flavor,
        dropout_rate=dropout_rate,
        noise_rate=noise_rate,
        ran_flip_conf=ran_flip_conf
    )

    # get autoencoder layers
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    # defining models
    autoencoder = Sequential(autoencoder_layers, name='AE')
    encoder = Sequential(encoder_layers, name='encoder')
    decoder = Sequential(decoder_layers, name='decoder')

    return (autoencoder, encoder, decoder)


def get_encoder_layers(
    dims: List[int] = None,
    activation='relu',
    w_init='glorot_uniform',
    u_norm_reg: bool = False,
    ortho_w_con: bool = False,
    uncoll_feat_reg: bool = False,
    use_bias: bool = True,
):
    encoder_layers = []
    for i in range(len(dims)):
        act = activation
        k_reg = WeightsOrthogonalityConstraint(
            dims[i], weightage=1., axis=0) if ortho_w_con else None
        k_con = UnitNorm(axis=0) if u_norm_reg else None
        act_reg = None
        if i == len(dims)-1:
            act = None
            act_reg = UncorrelatedFeaturesConstraint(
                dims[i], weightage=1.) if uncoll_feat_reg else None
        x = Dense(
            units=dims[i],
            activation=act,
            kernel_regularizer=k_reg,
            kernel_initializer=w_init,
            kernel_constraint=k_con,
            activity_regularizer=act_reg,
            use_bias=use_bias,
            name=encoder_layer_name % i)
        encoder_layers.append(x)
    return encoder_layers


def get_decoder_layers(
    dims: List[int] = None,
    activation='relu',
    is_tied: bool = True,
    w_init='glorot_uniform',
    u_norm_reg: bool = False,
    ortho_w_con: bool = False,
    use_bias: bool = True,
    output_activation='sigmoid',
    encoder_layers=None,
):
    decoder_layers = []
    if is_tied:
        for i in range(len(dims)):
            act = activation
            if i == len(dims)-1:
                act = output_activation
            x = DenseTied(
                tied_to=encoder_layers[len(encoder_layers)-1-i],
                units=dims[i],
                activation=act,
                kernel_initializer=w_init,
                use_bias=use_bias,
                name=decoder_layer_name % i)
            decoder_layers.append(x)
    else:
        for i in range(len(dims)):
            act = activation
            k_reg = WeightsOrthogonalityConstraint(
                dims[i], weightage=1., axis=1) if ortho_w_con else None
            k_con = UnitNorm(axis=1) if u_norm_reg else None
            if i == len(dims)-1:
                act = output_activation
            x = Dense(
                units=dims[i],
                activation=act,
                kernel_regularizer=k_reg,
                kernel_initializer=w_init,
                kernel_constraint=k_con,
                use_bias=use_bias,
                name=decoder_layer_name % i)
            decoder_layers.append(x)
    return decoder_layers


def get_output_act(flavor):
    act_dict = {
        'real': 'relu',
        'binary': 'sigmoid',
        'probability': 'sigmoid',
    }
    return act_dict[flavor]


def noise_layers_fn(
    layers,
    flavor: str = 'real',
    dropout_rate: float = 0.0,
    noise_rate: float = 0.0,
    ran_flip_conf: Dict = None,
):
    if dropout_rate > 0.0 and len(layers) > 2:
        idx = np.arange(start=2, stop=int((2*len(layers))-2), step=2)
        for i in idx:
            layers.insert(i, Dropout(rate=dropout_rate))
    if noise_rate > 0.0:
        layers.insert(1, get_noise_layer(flavor, noise_rate, ran_flip_conf))
    return layers


def get_noise_layer(
    flavor: str = 'real',
    noise_rate: float = 0.0,
    ran_flip_config=None
):
    noise_layer_dict = {
        'real': GaussianNoise(stddev=noise_rate),
        'binary': FlippingNoise(
            up_frequencies=ran_flip_config['up_frequencies'],
            b_idx=ran_flip_config['b_idx'],
            rate=noise_rate
        ) if ran_flip_config is not None else None,
        'probability': TruncatedGaussianNoise(
            stddev=noise_rate
        ),
    }
    return noise_layer_dict[flavor]


def create_clustering_model(n_clusters, encoder, alpha: float = 1.0):
    clustering_layer = ClusteringLayer(
        n_clusters, name='clustering', alpha=alpha)(encoder.output)
    return Model(inputs=encoder.input, outputs=clustering_layer)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
