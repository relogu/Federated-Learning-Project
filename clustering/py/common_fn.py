#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:25:15 2021

@author: relogu
"""

import math
import pathlib
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
from flwr.common import Weights
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import datasets
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             homogeneity_score, normalized_mutual_info_score,
                             rand_score)
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, InputSpec, Layer
from tensorflow.keras.models import Model
from torch.utils.data import Dataset
import lifelines
from lifelines import KaplanMeierFitter, WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, PiecewiseExponentialFitter, GeneralizedGammaFitter, SplineFitter


from py.dataset_util import plot_points_2d


def create_autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act,
                  kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' %
                    (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act,
                  kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


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


def create_keras_model():
    """Define the common keras model used.

    Returns:
        tensorflow.keras.models.Sequential: a simple two layer sequential model use for the toy dataset make_moons
    """
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(4, activation='tanh',
                              kernel_initializer=initializer,
                              bias_initializer='ones',),
        tf.keras.layers.Dense(2, activation='softmax',
                              kernel_initializer=initializer,
                              bias_initializer='ones',)])


def create_model():
    """Define the common keras model used.

    Returns:
        tensorflow.keras.models.Sequential: a simple two layer sequential model use for the toy dataset make_moons
    """
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((784,)),
        tf.keras.layers.Dense(4, activation='tanh',
                              kernel_initializer=initializer,
                              bias_initializer='ones',),
        tf.keras.layers.Dense(10, activation='softmax',
                              kernel_initializer=initializer,
                              bias_initializer='ones',)])


def dump_learning_curve(filename: str, round: int, loss: float, accuracy: float):
    """Dump the learning curve.
    The function dumps to the file given by complete path
    (relative or absolute) the row composed by:
    filename,round,loss,accuracy
    If round == 1, the function dumps also the header:
    \"client,round,loss,accuracy\"

    Args:
        filename ([str]): path to file to dump
        round ([int]): current round of the learning
        loss ([float]): current loss of the learning
        accuracy ([float]): current accuracy of the learning
    """
    # get file path
    path_to_file = pathlib.Path(__file__).parent.absolute()
    path_to_file = path_to_file/"output"/(filename+".dat")
    # touching file
    path_to_file.touch()
    with open(path_to_file, "a") as outfile:
        # write line(s)
        if round == 1:
            print("client,round,loss,accuracy", file=outfile)
        print(filename+","+str(round)+","+str(loss) +
              ","+str(accuracy), file=outfile)


def dump_result_dict(filename: str,
                     result: Dict,
                     path_to_out: Union[Path, str] = None,
                     verbose: int = 0):
    """Dump the result dictionary.
    The function dumps to the file given by complete path
    (relative or absolute) the row composed by results.values(),
    separated by a comma
    If result[\'round\'] == 1, the function dumps also the headers of 
    the dictionary, contained in results.keys(), separated by a comma.

    Args:
        filename ([str]): path to file to dump
        result ([Dict]): dictionary containing the values to dump
    """
    if result.get('round') is None:
        raise KeyError("The mandatory key \'round\' is missing.")
    # get file path
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()
    path_to_file = path_to_out/"output"/(filename+".dat")
    # touching file
    path_to_file.touch()
    if verbose > 0:
        print("Dumping results at "+str(path_to_file))
    with open(path_to_file, "a") as outfile:
        # write line(s)
        if result['round'] == 1:
            print(','.join(list(result.keys())), file=outfile)
        print(','.join(map(str, list(result.values()))), file=outfile)


def dump_pred_dict(filename: str,
                   pred: Dict,
                   path_to_out: Union[Path, str] = None,
                   verbose: int = 0):
    # get file path
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()
    path_to_file = path_to_out/"output"/(filename+".csv")
    if verbose > 0:
        print("Dumping results at "+str(path_to_file))
    df = pd.DataFrame(pred)
    df.to_csv(path_to_file)


def plot_lifelines_pred(time,
                        event,
                        labels,
                        fed_iter = None,
                        client_id = None,
                        path_to_out: Union[Path, str] = None):
    # setting path for saving image
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()
    path_to_file = path_to_out/"output"
    # initialize graph
    fig, axes = plt.subplots(1, 1, figsize=(25, 15))
    # setting title and filename
    if fed_iter is None and client_id is None:
        fig.suptitle("Final lifelines for the test set", fontsize=16)
        filename = 'lifelines_pred.png'
    elif fed_iter is None and client_id is not None:
        fig.suptitle("Actual lifelines for the test set for client {}". \
            format(client_id), fontsize=16)
        filename = 'lifelines_pred_'+str(client_id)+'.png'
    elif fed_iter is not None and client_id is None:
        fig.suptitle(
            "Lifelines for the test set at the federated round {}". \
                format(str(fed_iter)), fontsize=16)
        filename = 'lifelines_pred_e'+str(fed_iter)+'.png'
    else:
        fig.suptitle(
            "Lifelines for the test set at the federated round {} for client {}". \
                format(str(fed_iter), client_id), fontsize=16)
        filename = 'lifelines_pred_'+str(client_id)+'_e'+str(fed_iter)+'.png'
    # selected fitters
    fitters = {'KaplanMeierFitter': KaplanMeierFitter(),
               #'WeibullFitter': WeibullFitter(),
               #'ExponentialFitter': ExponentialFitter(),
               #'LogNormalFitter': LogNormalFitter(),
               #'LogLogisticFitter': LogLogisticFitter(),
               #'PiecewiseExponentialFitter': PiecewiseExponentialFitter([40, 60]),
               #'GeneralizedGammaFitter': GeneralizedGammaFitter()
               #'SplineFitter': SplineFitter(T.loc[E.astype(bool)], [0, 50, 100])
               }
    # loop on fitters
    i=j=0
    for key in fitters:
        ax = axes#[i][j]
        # loop on labels
        for label in np.unique(labels):
            idx = (labels == label)
            if len(idx) > 5:
                fitters[key].fit(time[idx], event[idx], label='f_{} l_{}'.format(key, label))
                fitters[key].plot_survival_function(ax=ax)
        i+=1
        if i>1:
            i=0
            j+=1
    # plt.show(block=False)
    # dump to a file
    plt.savefig(path_to_file/filename)
    plt.close()


def plot_dec_bound(model, x):
    """Plot the decision boundaries given by model.
    The vector x is used only to set the range of the axis.

    Args:
        model (tensorflow.keras.Model): model from which get the predictions
        x (ndarray of shape (n_samples, 2)): vector of 2-D points to plot

    Returns:
        (matplotlib.pyplot.QuadContourSet)
    """
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    return plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)


def plot_decision_boundary(model, x_test, y_test, client_id=None, fed_iter=None, path=None):
    """Plot the decision boundary given the predictions of the model.

    Args:
        model (tensorflow.keras.Model): model from which get the predictions
        x_test (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the train set
        y_test (ndarray of shape (n_samples)): vector of numerical labels for the train set
        client_id (str or int or cast to str, optional): identifier for the client. Defaults to None.
        fed_iter (int, optional): current federated step of building title. Defaults to None.
        path (str, optional): complete or relative path to output folder. Defaults to None.
    """
    # setting path for saving image
    if path is None:
        path = 'output'
    # initialize graph
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    if fed_iter is None and client_id is None:
        ax.set_title("Final decision boundary for the test set")
    else:
        ax.set_title(
            "Decision boundary for the test set at the federated round: " + str(fed_iter))
    if client_id is None:
        title = 'Decison boundary for aggregated model'
    else:
        title = 'Decison boundary for client-'+str(client_id)+' model'
    if fed_iter is not None:
        title += ' at iteration '+str(fed_iter)
    ax.set_title(title)
    # plot dec boundary
    plot_dec_bound(model, x_test)
    # plot test points
    plot_points_2d(x_test, y_test)
    plt.draw()
    # plt.show(block=False)
    # dump to a file
    if client_id is None:
        filename = path+'/dec_bound_nofed'
    else:
        filename = path+'/dec_bound_c'+str(client_id)
    if fed_iter is None:
        filename += '.png'
    else:
        filename += '_e'+str(fed_iter)+'.png'
    plt.savefig(filename)
    plt.close()


def print_confusion_matrix(y,
                           y_pred,
                           client_id=None,
                           fed_iter=None,
                           path_to_out: Union[Path, str] = None):
    # setting path for saving image
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()
    path_to_file = path_to_out/"output"
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    # dump to a file
    if client_id is None:
        filename = 'conf_matrix_nofed'
    else:
        filename = ('conf_matrix_c'+str(client_id))
    if fed_iter is None:
        filename += '.png'
    else:
        filename += '_e'+str(fed_iter)+'.png'
    plt.savefig(path_to_file/filename)
    plt.close()
