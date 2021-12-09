#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
from typing import List, Dict
import numpy as np

from tensorflow.keras.layers import InputLayer, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.constraints import UnitNorm

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


def create_autoencoder(net_arch, up_frequencies):
    if net_arch['binary']:
        return create_denoising_autoencoder(
            flavor='probability',
            dims=net_arch['dims'],
            activation=net_arch['act'],
            w_init='glorot_uniform',
            is_tied=net_arch['tied'],
            u_norm_reg=net_arch['u_norm'],
            ortho_w_con=net_arch['ortho'],
            uncoll_feat_reg=False,
            use_bias=True,
            dropout_rate=net_arch['dropout'],
            noise_rate=net_arch['ran_flip'],
            noise_conf_dict={'b_idx': net_arch['b_idx'],
                             'up_freq': up_frequencies}
        )
    else:
        return create_denoising_autoencoder(
            flavor='real',
            dims=net_arch['dims'],
            activation=net_arch['act'],
            w_init='glorot_uniform',
            is_tied=net_arch['tied'],
            u_norm_reg=net_arch['u_norm'],
            ortho_w_con=net_arch['ortho'],
            uncoll_feat_reg=False,
            use_bias=True,
            dropout_rate=net_arch['dropout'],
            noise_rate=net_arch['ran_flip'],
            noise_conf_dict={'stddev': net_arch['noise_stddev']}
        )


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
    noise_conf_dict: Dict = None,
):
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    # input data
    input_data = InputLayer(input_shape=(dims[0],), name='input_data')
    # input features
    input_feat = InputLayer(input_shape=(dims[-1],), name='input_feat')

    # build encoder layers
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
        noise_conf_dict=noise_conf_dict
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
    noise_conf_dict: Dict = None,
):
    if dropout_rate > 0.0 and len(layers) > 2:
        idx = np.arange(start=2, stop=int((2*len(layers))-2), step=2)
        for i in idx:
            layers.insert(i, Dropout(rate=dropout_rate))
    if noise_rate > 0.0:
        layers.insert(1, get_noise_layer(flavor, noise_rate, noise_conf_dict))
    return layers


def get_noise_layer(
    flavor: str = 'real',
    noise_rate: float = 0.0,
    noise_conf_dict=None
):
    noise_layer_dict = {
        'real': GaussianNoise(stddev=noise_conf_dict['stddev']),
        'binary': FlippingNoise(
            up_frequencies=noise_conf_dict['up_frequencies'],
            b_idx=noise_conf_dict['b_idx'],
            rate=noise_rate
        ) if noise_conf_dict is not None else None,
        'probability': TruncatedGaussianNoise(
            stddev=noise_conf_dict['stddev'],
            rate=noise_rate
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
