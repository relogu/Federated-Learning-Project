#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 10:29:15 2021

@author: relogu
"""
import argparse
import pathlib
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Dense, Input, InputSpec, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import UnitNorm
from sklearn.cluster import KMeans

path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(path.parent))

from py.util import target_distribution
import py.dataset_util as data_util
import clustering.py.common_fn as my_fn
import py.metrics as my_metrics


out_1 = 'UDE for Clustering\nEpoch %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f'


def get_parser():
    parser = argparse.ArgumentParser(description="UDE Training Script")
    parser.add_argument("--batch_size", dest="batch_size",
                        default=64, type=int, help="Batch size")
    parser.add_argument("--hardware_acc", dest="cuda_flag", action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("--lim_cores", dest="lim_cores",
                        action='store_true', help="Flag for limiting cores")
    parser.add_argument("--folder", dest="out_folder",
                        type=type(str('')), help="Folder to output images")
    parser.add_argument('--groups', dest='groups',
                        required=False,
                        type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7],
                        default=7,
                        action='store',
                        help='how many groups of variables to use for EUROMDS dataset')
    parser.add_argument("--n_clusters", dest="n_clusters", default=10,
                        type=int, help="Define the number of clusters to identify")
    parser.add_argument('--fold_n',
                        dest='fold_n',
                        required=False,
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        action='store',
                        help='fold number for train-test partitioning')
    parser.add_argument('--shuffle',
                        dest='shuffle',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='wheater to shuffle in train-test partitioning')
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=True,
                        type=int,
                        default=200,
                        action='store',
                        help='number of epochs for the autoencoder pre-training')
    parser.add_argument('--cl_epochs',
                        dest='cl_epochs',
                        required=True,
                        type=int,
                        default=1000,
                        action='store',
                        help='number of epochs for the clustering step')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    return parser



class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 weights=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.initial_weights = weights
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            #self.kernel = K.transpose(self.tied_to.kernel)
            self.kernel = self.tied_to.kernel
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, K.transpose(self.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()))
    

class WeightsOrthogonalityConstraint(constraints.Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)
    
class UncorrelatedFeaturesConstraint(constraints.Constraint):
        
    def __init__(self, encoding_dim, weightage = 1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
    
    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))
        
        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)
        
        return covariance
            
    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = K.sum(K.square(
                self.covariance - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)
    

def create_autoencoder(dims, act='relu', init='glorot_uniform', verbose=False):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    # getting encoder and decoder layers output dim
    encoder_dims = dims[1:]
    decoder_dims = list(reversed(dims))[1:]
    if verbose:
        print('List of encoder dimensions: {}'.format(encoder_dims))
        print('List of decoder dimensions: {}'.format(decoder_dims))
    # input data
    input_img = Input(shape=(dims[0],), name='input_img')
    # input labels
    input_lbl = Input(shape=(dims[-1],), name='input_lbl')
    ## encoder
    encoder_layers = []
    encoder_layers.append(input_img)
    # internal layers in encoder
    for i in range(len(encoder_dims)):
        x = Dense(units=encoder_dims[i],
                  activation=act,
                  #kernel_regularizer=WeightsOrthogonalityConstraint(encoder_dims[i], weightage=1., axis=0),
                  kernel_initializer=init,
                  #kernel_constraint=UnitNorm(axis=0),
                  use_bias=False,#True,
                  name='encoder_%d' % i)
        if verbose:
            print('Encoder Layer {}: {} with dims {}'.format('encoder_%d' % i, x, encoder_dims[i]))
        encoder_layers.append(x)
    

    ## decoder
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
                      #kernel_initializer=init,
                      use_bias=False,#True,
                      #kernel_constraint=UnitNorm(axis=1),
                      name='decoder_%d' % i)
        if verbose:
            print('Decoder Layer {}: {} with dims {}, tied to {}'.format('decoder_%d' % i, x, decoder_dims[i], encoder_layers[len(encoder_layers)-1-i]))
        decoder_layers.append(x)

    
    ## autoencoder
    autoencoder_layers = []
    autoencoder_layers = autoencoder_layers + encoder_layers
    autoencoder_layers = autoencoder_layers + decoder_layers[1:]

    ## defining models
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


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` which represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(
            self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        # Make sure each sample's 10 values add up to 1.
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    # configuration
    # get parameters
    args = get_parser().parse_args()
    # disable possible gpu devices (add hard acc, selection)
    if not args.cuda_flag:
        print('No CUDA')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif args.lim_cores:
        print('Limiting CPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    # defining output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    os.makedirs(path_to_out, exist_ok=True)
    # initializing common configuration dict
    config = {
        'batch_size': args.batch_size,
        'splits': 5,
        'fold_n': args.fold_n,
        'n_clusters': args.n_clusters,
        'shuffle': args.shuffle,
        'kmeans_epochs': 300,
        'kmeans_n_init': 25,
        'ae_epochs': args.ae_epochs,
        'ae_lr': 0.01,
        'ae_momentum': 0.9,
        'cl_lr': 0.01,
        'cl_momentum': 0.9,
        'cl_epochs': args.cl_epochs,
        'update_interval': 55,
        'ae_loss': 'mse',#'binary_crossentropy',#'mse',
        'cl_loss': 'kld',
        'seed': args.seed}

    # preparing dataset
    groups = ['Genetics', 'CNA', 'GeneGene', 'CytoCyto', 'GeneCyto', 'Demographics', 'Clinical']
    # getting the entire dataset
    x = data_util.get_euromds_dataset(groups=groups[:args.groups])
    # getting the number of features
    n_features = len(x.columns)
    x = np.array(x)
    # getting labels from HDP
    prob = data_util.get_euromds_dataset(groups=['HDP'])
    y = []
    for label, row in prob.iterrows():
        if np.sum(row) > 0:
            y.append(row.argmax())
        else:
            y.append(-1)
    y = np.array(y)
    # getting the outcomes
    outcomes = data_util.get_outcome_euromds_dataset()
    outcomes = np.array(outcomes[['outcome_3', 'outcome_2']])
    # getting IDs
    ids = data_util.get_euromds_ids()
    # setting the autoencoder layers
    #dims = [x.shape[-1], x.shape[-1], int((n_features+args.n_clusters)/2), int((n_features+args.n_clusters)/2), args.n_clusters]
    #dims = [x.shape[-1], x.shape[-1], int((n_features+args.n_clusters)/2), int((n_features+args.n_clusters)/2), int((n_features+args.n_clusters)/2), args.n_clusters]
    dims = [x.shape[-1], int((n_features+args.n_clusters)/2), int((n_features+args.n_clusters)/2), args.n_clusters]
    
    config['ae_lr'] = 0.1
    config['ae_dims'] = dims
    # define the splitting
    train_idx, test_idx = data_util.split_dataset(
        x=x,
        splits=config['splits'],
        shuffle=config['shuffle'],
        fold_n=config['fold_n'],
        r_state=config['seed'])
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train, y_test = None, None
    outcomes_train, outcomes_test = None, None
    id_train, id_test = None, None
    if y is not None:
        y_test = y[test_idx]
        y_train = y[train_idx]
    if outcomes is not None:
        outcomes_train = outcomes[train_idx]
        outcomes_test = outcomes[test_idx]
    if ids is not None:
        id_train = ids[train_idx]
        id_test = ids[test_idx]

    # pre-train the autoencoder
    autoencoder, encoder, decoder = create_autoencoder(
        config['ae_dims'], act='relu')#, init='glorot_normal')#, act='linear')
    ae_optimizer = SGD( lr=config['ae_lr'], decay=(config['ae_lr']-0.001)/config['ae_epochs'], momentum=config['ae_momentum'])
    autoencoder.compile(
        metrics=['accuracy'],
        optimizer=ae_optimizer,
        loss=config['ae_loss']
    )
    # fitting the autoencoder
    #for i in range(int(config['ae_epochs'])):
    autoencoder.fit(x=x_train,
                    y=x_train,
                    batch_size=config['batch_size'],
                    epochs=int(config['ae_epochs']),
                    verbose=1)
    # evaluation of the autoencoder
    loss = autoencoder.evaluate(
        x_test, x_test, verbose=2)
    '''
    metrics = {"loss": loss}
    result = metrics.copy()
    result['round'] = i+1
    my_fn.dump_result_dict('pretrain_ae', result,
                            path_to_out=path_to_out)'''

    # get an estimate for clusters centers using k-means
    kmeans = KMeans(init='k-means++',
                    n_clusters=config['n_clusters'],
                    # number of different random initializations
                    n_init=config['kmeans_n_init'],
                    random_state=config['seed'])
    # fitting clusters' centers using k-means
    kmeans.fit(encoder.predict(x_train))

    # training the clustering model
    clustering_model = create_clustering_model(
        config['n_clusters'],
        encoder)
    # compiling the clustering model
    cl_optimizer = SGD(
        lr=config['cl_lr'], momentum=config['cl_momentum'])
    clustering_model.compile(
        optimizer=cl_optimizer,
        loss=config['cl_loss'])
    clustering_model.get_layer(
        name='clustering').set_weights(np.array([kmeans.cluster_centers_]))
    for i in range(int(config['cl_epochs'])):
        if i % config['update_interval'] == 0:
            q = clustering_model.predict(x_train, verbose=0)
            # update the auxiliary target distribution p
            p = target_distribution(q)
        clustering_model.fit(x=x_train, y=p, verbose=2,
                             batch_size=config['batch_size'])
        # evaluation
        q_eval = clustering_model.predict(x_test, verbose=0)
        # update the auxiliary target distribution p
        p_eval = target_distribution(q_eval)
        # retrieving loss
        loss = clustering_model.evaluate(x_test, p_eval, verbose=2)
        # evaluate the clustering performance using some metrics
        y_pred = q_eval.argmax(1)
        # evaluating metrics
        if y_test is not None:
            acc = my_metrics.acc(y_test, y_pred)
            nmi = my_metrics.nmi(y_test, y_pred)
            ami = my_metrics.ami(y_test, y_pred)
            ari = my_metrics.ari(y_test, y_pred)
            ran = my_metrics.ran(y_test, y_pred)
            homo = my_metrics.homo(y_test, y_pred)
            if i % 10 == 0:  # print confusion matrix
                my_fn.print_confusion_matrix(
                    y_test, y_pred,
                    path_to_out=path_to_out)
            print(out_1 % (i+1, acc, nmi, ami, ari, ran, homo))
            # dumping and retrieving the results
            metrics = {"accuracy": acc,
                       "normalized_mutual_info_score": nmi,
                       "adjusted_mutual_info_score": ami,
                       "adjusted_rand_score": ari,
                       "rand_score": ran,
                       "homogeneity_score": homo}
            result = metrics.copy()
            result['loss'] = loss
            result['round'] = i+1
            my_fn.dump_result_dict('clustering_model', result,
                                   path_to_out=path_to_out)
        if id_test is not None:
            pred = {'ID': id_test,
                    'label': y_pred}
            my_fn.dump_pred_dict('pred', pred,
                                 path_to_out=path_to_out)
    '''
    # freeze the encoder to train the decoder to be used then
    encoder.trainable = False
    # re-compiling to fiz the freezing
    autoencoder.compile(
        optimizer=ae_optimizer,
        loss=config['ae_loss']
    )
    
    # fitting the autoencoder again
    for i in range(int(config['ae_epochs'])):
        autoencoder.fit(x=x_train,
                        y=x_train,
                        batch_size=config['batch_size'],
                        verbose=2)
        # evaluation of the autoencoder
        loss = autoencoder.evaluate(
            x_test, x_test, verbose=2)
        metrics = {"loss": loss}
        result = metrics.copy()
        result['round'] = i+1
        my_fn.dump_result_dict('retrain_ae', result,
                               path_to_out=path_to_out)'''
    
    parameters = np.array(clustering_model.get_weights(), dtype=object)
    np.savez(path_to_out/'clustering_model', parameters)
    parameters = np.array(decoder.get_weights(), dtype=object)
    np.savez(path_to_out/'decoder', parameters)
    parameters = np.array(autoencoder.get_weights(), dtype=object)
    np.savez(path_to_out/'autoencoder', parameters)
    parameters = np.array(encoder.get_weights(), dtype=object)
    np.savez(path_to_out/'encoder', parameters)

