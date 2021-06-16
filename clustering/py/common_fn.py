#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:25:15 2021

@author: relogu
"""

from typing import List, Optional, Tuple, Dict
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import seaborn as sns
from flwr.common import Weights
import sklearn
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, rand_score, homogeneity_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment

# definition of the metrics used
nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score
ran = rand_score
homo = homogeneity_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def simple_clustering_on_fit_config(rnd: int,
                                    ae_epochs: int = 300,
                                    kmeans_epochs: int = 20,
                                    cl_epochs: int = 1000):
    if rnd < ae_epochs+1:
        return {'model': 'autoencoder',
                'first': (rnd==1),
                'actual_round': rnd,
                'total_round': ae_epochs}
    elif rnd < ae_epochs+kmeans_epochs+1:
        return {'model': 'k-means',
                'first': (rnd==ae_epochs+1),
                'actual_round': rnd-ae_epochs,
                'total_round': kmeans_epochs}
    else:
        return {'model': 'clustering',
                'first': (rnd==ae_epochs+kmeans_epochs+1),
                'actual_round': rnd-ae_epochs-kmeans_epochs,
                'total_round': cl_epochs}


def kfed_clustering_on_fit_config(rnd: int,
                                  ae_epochs: int = 300,
                                  cl_epochs: int = 1000,
                                  n_clusters: int = 2):
    if rnd < ae_epochs+1:
        config = {'model': 'pretrain_ae',
                  'n_clusters': n_clusters,
                  'first': (rnd == 1),
                  'actual_round': rnd-1,
                  'total_rounds': ae_epochs}
    elif rnd < ae_epochs+2:
        config = {'model': 'k-FED',
                  'n_clusters': n_clusters,
                  'first': (rnd ==  ae_epochs+1),
                  'actual_round': rnd,
                  'total_rounds': 1}
    else:
        config = {'model': 'clustering',
                  'n_clusters': n_clusters,
                  'first': (rnd == ae_epochs+2),
                  'actual_round': rnd-ae_epochs-1,
                  'total_rounds': cl_epochs}
    return config

def simple_kmeans_on_fit_config(rnd: int,
                                kmeans_epochs: int = 20):
    if rnd < kmeans_epochs+1:
        return {'model': 'k-means'}

def distance_from_centroids(centroids_array, vector):
    distances = []
    for centroid in centroids_array:
        d = np.linalg.norm(centroid-vector)
        distances = np.append(distances, d)
    return min(distances)

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
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def create_clustering_model(n_clusters, encoder):
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    return Model(inputs=encoder.input, outputs=clustering_layer)

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

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
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
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
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
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
        print(filename+","+str(round)+","+str(loss)+","+str(accuracy), file=outfile)

def dump_result_dict(filename: str, result: Dict, verbose: int = 0):
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
    path_to_file = pathlib.Path(__file__).parent.parent.absolute()
    path_to_file = path_to_file/"output"/(filename+".dat")
    # touching file
    path_to_file.touch()
    if verbose > 0:
        print("Dumping results at "+str(path_to_file))
    with open(path_to_file, "a") as outfile:
        # write line(s)
        if result['round'] == 1:
            print(','.join(list(result.keys())), file=outfile)
        print(','.join(map(str,list(result.values()))), file=outfile)

def translate_moons(dx: float, dy: float, x):
    """Translate using the vector (dx, dy) the make_moons dataset x.
    The function will retrieve a copy of x.

    Args:
        dx (float): x-component of the translation vector
        dy (float): y-component of the translation vector
        x (ndarray of shape (n_samples, 2)): list of 2-D points generated by sklearn.datasets.make_moons()

    Returns:
        (ndarray of shape (n_samples, 2)): translated list of 2-D points generated by sklearn.datasets.make_moons()
    """
    # get a copy
    xc = x.copy()
    # check on shape
    if x.shape[1] == 2:
        # applying transformation
        xc[:, 0] = x[:, 0] + dx
        xc[:, 1] = x[:, 1] + dy
    else :
        # error msg
        raise TypeError("the input x has not the correct shape")
    return xc

def rotate_moons(theta: float, x):
    """Rotate using the angle theta the make_moons dataset x w.r.t the origin (0,0).
    The function will retrieve a copy of x.

    Args:
        theta (float): angle generator for the rotation transformation
        x (ndarray of shape (n_samples, 2)): list of 2-D points generated by sklearn.datasets.make_moons()

    Returns:
        (ndarray of shape (n_samples, 2)): rotated list of 2-D points generated by sklearn.datasets.make_moons()
    """
    # get a copy
    xc = x.copy()
    # check on shape
    if xc.shape[1] == 2:
        # applying tranformation
        xc[:, 0] = x[:, 0]*math.cos(theta) - x[:, 1]*math.sin(theta)
        xc[:, 1] = x[:, 0]*math.sin(theta) + x[:, 1]*math.cos(theta)
    else :
        # error msg
        raise TypeError("the input x has not the correct shape")
    return xc

def plot_points(x, y):
    """Plot the points x coloring them by the labels in vector y

    Args:
        x (ndarray of shape (n_samples, 2)): vector of 2-D points to plot
        y (ndarray of shape (n_samples)): vector of numerical labels
    
    Returns:
        (matplotlib.pyplot.PathCollection)
    """
    return plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)

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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    return plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)    

def plot_client_dataset(client_id, x_train, y_train, x_test, y_test, path=None):
    """Plot and dump to a file the data samples given the specified client id and dataset.

    Args:
        client_id (str or int or cast to str): identifier for the client
        x_train (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the train set
        y_train (ndarray of shape (n_samples)): vector of numerical labels for the train set
        x_test (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the test set
        y_test (ndarray of shape (n_samples)): vector of numerical labels for the test set
    """
    # setting path for saving image
    if path is None:
        path = 'output'
    # initialize graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9))
    ax.set_title("Data samples for the client " + str(client_id))
    ax.set_xlabel('x')
    ax.set_ylabel('Y')
    # Plot the samples
    plot_points(x_train, y_train)
    # augment test to be colored differently
    y_test = y_test+2
    plot_points(x_test, y_test)
    plt.draw()
    #plt.show(block=False)
    plt.savefig(path+'/data_client_'+str(client_id)+'.png')
    plt.close()

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
    if fed_iter is None and client_id is None: ax.set_title("Final decision boundary for the test set")
    else: ax.set_title("Decision boundary for the test set at the federated round: " + str(fed_iter))
    if client_id is None: title = 'Decison boundary for aggregated model'
    else: title = 'Decison boundary for client-'+str(client_id)+' model'
    if fed_iter is not None: title += ' at iteration '+str(fed_iter)  
    ax.set_title(title)
    # plot dec boundary
    plot_dec_bound(model, x_test)
    # plot test points
    plot_points(x_test, y_test)
    plt.draw()
    #plt.show(block=False)
    # dump to a file
    if client_id is None: filename = path+'/dec_bound_nofed'
    else: filename = path+'/dec_bound_c'+str(client_id)
    if fed_iter is None: filename += '.png'
    else: filename += '_e'+str(fed_iter)+'.png'    
    plt.savefig(filename)
    plt.close()
    
def print_confusion_matrix(y, y_pred, client_id=None, fed_iter=None, path=None):
    # setting path for saving image
    if path is None:
        path = 'output'
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    # dump to a file
    if client_id is None: filename = path+'/conf_matrix_nofed'
    else: filename = path+'/conf_matrix_c'+str(client_id)
    if fed_iter is None: filename += '.png'
    else: filename += '_e'+str(fed_iter)+'.png'    
    plt.savefig(filename)
    plt.close()

def build_dataset(n_clients: int, total_samples: int, noise: float, seed: int=51550):
    """Build the entire dataset, to be distributed.

    Args:
        n_clients (int): number of clients onto which distribute the whole dataset
        total_samples (int): total number of sample of the whole datset
        noise (float): the amount of noise to generate the dataset
        seed (int): the seed for the generator of the dataset

    Returns:
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels
    """
    # getting the number of samples of the clients' dataset
    N_SAMPLES = int(total_samples/n_clients)
    # initializing arrays of points and labels (may be not needed)
    x = np.array(0)
    y = np.array(0)
    # set the intial seed for the RN generator
    random.seed(seed)
    # loop on clients
    for i in range(n_clients):
        # get a RN for the state of the dataset generator
        train_rand_state = random.randint(0, 100000)
        # get data points and labels
        (x_client, y_client) = datasets.make_moons(n_samples=int(N_SAMPLES),
                                                   noise=noise,
                                                   shuffle=True,
                                                   random_state=train_rand_state)
        # fill the arrays of points and labels
        if i == 0:
            x = x_client
            y = y_client
        else :
            x = np.concatenate((x, x_client), axis=0)
            y = np.concatenate((y, y_client), axis=0)       
    return x, y

def build_mnist_dataset(n_clients: int, total_samples: int, noise: float, seed: int=51550):
    """Build the entire dataset, to be distributed.

    Args:
        n_clients (int): number of clients onto which distribute the whole dataset
        total_samples (int): total number of sample of the whole datset
        noise (float): the amount of noise to generate the dataset
        seed (int): the seed for the generator of the dataset

    Returns:
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels
    """
    # getting the number of samples of the clients' dataset
    N_SAMPLES = int(total_samples/n_clients)
    # initializing arrays of points and labels (may be not needed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x[0:N_SAMPLES], y[0:N_SAMPLES]

def get_client_dataset(client_id: int, n_clients: int, x_tot, y_tot):
    """Get the single client dataset given the whole dataset.

    Args:
        client_id (int): identifier of the client, must be inside [0, (n_clients) - 1]
        n_clients (int): number of clients onto which the whole dataset is being distributed
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels

    Returns:
        x (ndarray of shape (single_client_samples, 2)): vector of 2-D points relative to the client
        y (ndarray of shape (single_client_samples)): vector of numerical labels relative to the client
    """
    # check on the shapes of the inputs
    if client_id >= n_clients or client_id < 0:
        msg = "the input client_id has not an allowed value, " + \
            "insert a positive value lesser than n_clients"
        raise TypeError(msg)
    if len(x_tot.shape) != 2 or x_tot.shape[1] != 2:
        msg = "the input x_tot has not the correct shape"
        raise TypeError(msg)
    if len(y_tot.shape) != 1:
        msg = "the input y_tot has not the correct shape"
        raise TypeError(msg)
    if y_tot.shape[0] != x_tot.shape[0]:
        msg = "the inputs x_tot and y_tot have not compatible shapes, " + \
            "they must represent the same number of points"
        raise TypeError(msg)
    # get the total number of samples
    n_samples = x_tot.shape[0]
    # get the number of samples for the single clients
    n_sam_client = int(n_samples/n_clients)
    # loop on clients
    for i in range(n_clients):
        # continue on wrong clients and returning the right dataset
        if i != client_id:
            continue
        else:
            return x_tot[i*n_sam_client:(i+1)*n_sam_client], y_tot[i*n_sam_client:(i+1)*n_sam_client]

def get_client_mnist_dataset(client_id: int, n_clients: int, x_tot, y_tot):
    """Get the single client dataset given the whole dataset.

    Args:
        client_id (int): identifier of the client, must be inside [0, (n_clients) - 1]
        n_clients (int): number of clients onto which the whole dataset is being distributed
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels

    Returns:
        x (ndarray of shape (single_client_samples, 2)): vector of 2-D points relative to the client
        y (ndarray of shape (single_client_samples)): vector of numerical labels relative to the client
    """
    # check on the shapes of the inputs
    if client_id >= n_clients or client_id < 0:
        msg = "the input client_id has not an allowed value, " + \
            "insert a positive value lesser than n_clients"
        raise TypeError(msg)
    if len(x_tot.shape) != 2 or x_tot.shape[1] != 784:
        msg = "the input x_tot has not the correct shape"
        raise TypeError(msg)
    if len(y_tot.shape) != 1:
        msg = "the input y_tot has not the correct shape"
        raise TypeError(msg)
    if y_tot.shape[0] != x_tot.shape[0]:
        msg = "the inputs x_tot and y_tot have not compatible shapes, " + \
            "they must represent the same number of points"
        raise TypeError(msg)
    # get the total number of samples
    n_samples = x_tot.shape[0]
    # get the number of samples for the single clients
    n_sam_client = int(n_samples/n_clients)
    # loop on clients
    for i in range(n_clients):
        # continue on wrong clients and returning the right dataset
        if i != client_id:
            continue
        else:
            return x_tot[i*n_sam_client:(i+1)*n_sam_client], y_tot[i*n_sam_client:(i+1)*n_sam_client]
