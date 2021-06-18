#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import sys
import pathlib
import math
import random
import argparse
import os
import flwr as fl
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from flwr.dataset.utils.common import create_lda_partitions, create_partitions
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(path.parent.parent))
import clustering.py.common_fn as my_fn
import clustering.py.my_clients as clients
# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"


def parse_args():
    """Parse the arguments passed."""
    description = 'Client for moons test FL network using flower.\n' + \
        'Give the id of the client and the number of local epoch you want to perform.\n' + \
        'Give also the number of data samples you want to train for this client.\n' + \
        'Give also the number of clients in the FL set up to build properly the dataset.\n' + \
        'One can optionally give the server location to pass the client builder.\n' + \
        'One can optionally give the number of local epochs to perform.\n' + \
        'One can optionally give the noise to generate the dataset.\n' + \
        'One can optionally tell the program to plot the decision boundary at the evaluation step.\n' + \
        'One can optionally tell the program to use the shared test set (default) or the train set as test also.\n' + \
        'The client id will also initialize the seed for the train dataset.\n'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--client_id',
                        dest='client_id',
                        required=True,
                        type=int,
                        action='store',
                        help='client identifier')
    parser.add_argument('--dataset',
                        dest='dataset',
                        required=True,
                        type=type(''),
                        default='blobs',
                        choices=['blobs', 'moons', 'mnist'],
                        action='store',
                        help='client dataset identifier')
    parser.add_argument('--alg',
                        dest='alg',
                        required=True,
                        type=type(''),
                        default='k-means',
                        choices=['k-means', 'k_fed-ae_clust',
                                 'k-ae_clust', 'clustergan'],
                        action='store',
                        help='algorithm identifier')
    parser.add_argument('--n_samples',
                        dest='n_samples',
                        required=True,
                        type=int,
                        default=500,
                        action='store',
                        help='number of total samples in whole training set')
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        default=2,
                        choices=[2, 3, 4, 5, 6, 7, 8],
                        action='store',
                        help='number of total clients in the FL setting')
    parser.add_argument('--n_clusters',
                        dest='n_clusters',
                        required=True,
                        type=int,
                        default=2,
                        action='store',
                        help='number of total clusters to initialize the kMeans algorithm')
    parser.add_argument('--server',
                        dest='server',
                        required=False,
                        type=type(''),
                        default='[::]:51550',
                        action='store',
                        help='server address to point')
    parser.add_argument('--rounds',
                        dest='rounds',
                        required=False,
                        type=int,
                        default=1,
                        action='store',
                        help='number of local epochs to perform at each federated epoch')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('--noise',
                        dest='noise',
                        required=False,
                        type=float,
                        default=0.1,
                        action='store',
                        help='noise to put in the train dataset')
    parser.add_argument('--lda',
                        dest='lda',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='wheater to apply LDA partitioning to the entire dataset')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":

    # parsing arguments
    args = parse_args()

    # managing parameters
    CLIENT_ID = args.client_id
    DATASET = args.dataset
    N_SAMPLES = args.n_samples
    N_CLIENTS = args.n_clients
    N_CLUSTERS = args.n_clusters
    N_LOC_EPOCHS = args.rounds
    SEED = args.seed
    R_NOISE = args.noise
    USE_LDA = args.lda
    SERVER = args.server
    print("Server address: "+SERVER)

    # dataset, building the whole one and get the local
    if DATASET == 'blobs':
        n_features = 30
        X = datasets.make_blobs(
            n_samples=N_SAMPLES,
            n_features=n_features,
            random_state=SEED,
            centers=N_CLUSTERS)
        if USE_LDA:
            Y, _ = create_lda_partitions(dataset=X, num_partitions=N_CLIENTS)
        else:
            Y = create_partitions(unpartitioned_dataset=X,
                                  iid_fraction=0.5,
                                  num_partitions=N_CLIENTS)
        x, y = Y[CLIENT_ID]
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], int(4*n_features), N_CLUSTERS]
    elif DATASET == 'moons':
        x_tot, y_tot = my_fn.build_dataset(N_CLIENTS, N_SAMPLES, R_NOISE, SEED)
        x_train, y_train = my_fn.get_client_dataset(
            args.client_id, N_CLIENTS, x_tot, y_tot)
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], 8, 8, 32, N_CLUSTERS]
    elif DATASET == 'mnist':
        (X1, Y1), (X2, Y2) = tf.keras.datasets.mnist.load_data()
        X = np.concatenate((X1, X2))
        Y = np.concatenate((Y1, Y2))
        X = (X, Y)
        if USE_LDA:
            Y, _ = create_lda_partitions(dataset=X, num_partitions=N_CLIENTS)
        else:
            iid_frac = X[0].shape[0]/int((X[0].shape[0]/(N_CLIENTS*3))*0.5)
            Y = create_partitions(unpartitioned_dataset=X,
                                  iid_fraction=0.5,
                                  num_partitions=N_CLIENTS)
        x, y = Y[CLIENT_ID]
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 255.)
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], 500, 500, 2000, N_CLUSTERS]

    # initializing kmeans
    kmeans = KMeans(n_clusters=N_CLUSTERS,
                    max_iter=1,
                    verbose=0,
                    random_state=SEED)
    y_pred_kmeans = kmeans.fit_predict(x)
    print('Client %d, initial accuracy of k-Means: %.5f' %
          (CLIENT_ID, my_fn.acc(y, y_pred_kmeans)))

    '''
    # some hyperparameters
    UPDATE_INTERVAL = 140
    PRETRAIN_EPOCHS = 150
    AUTOENC_LOC_EPOCHS = 5
    # k-means plus autoencoder with clustering layer
    client = clients.ClusteringClient(autoencoder=autoencoder, encoder=encoder, kmeans=kmeans,
                                  clustering_model=my_fn.create_clustering_model(N_CLUSTERS, encoder),
                                  x=x, y=y, client_id=CLIENT_ID, n_clusters=N_CLUSTERS)
    # k-means plus autoencoder model
                    
    # autoencoder for clustering
    client = clients.SimpleClusteringClient(autoencoder=autoencoder, encoder=encoder,
                                            x=x, y=y, client_id=CLIENT_ID, n_clusters=N_CLUSTERS,
                                            ae_local_epochs=1)'''
    # algorithm choice
    if args.alg == 'k-means':
        client = clients.SimpleKMeansClient(x=x,
                                            y=y,
                                            seed=SEED,
                                            n_clusters=N_CLUSTERS,
                                            client_id=CLIENT_ID)
    elif args.alg == 'k_fed-ae_clust':
        # initializing hyperparameters and encoder
        autoencoder, encoder = my_fn.create_autoencoder(dims=dims)
        autoencoder.summary()
        client = clients.KFEDClusteringClient(x=x,
                                              y=y,
                                              ae_dims=dims,
                                              client_id=CLIENT_ID,
                                              n_clusters=N_CLUSTERS)
    elif args.alg == 'k-ae_clust':
        autoencoder, encoder = my_fn.create_autoencoder(dims=dims)
        autoencoder.summary()
        client = clients.CommunityClusteringClient(autoencoder,
                                                   encoder,
                                                   kmeans,
                                                   my_fn.create_model(),
                                                   x,
                                                   y,
                                                   CLIENT_ID,
                                                   ae_fed_epochs=1,
                                                   n_clusters=10,
                                                   local_epochs=1,
                                                   ae_local_epochs=300,
                                                   n_communities=2)
    elif args.alg == 'clustergan':
        config = {
            'batch_size': 32,
            'latent_dim': 30,
            'n_clusters': 10,
            'betan': 10,
            'betac': 10,
            'n_local_epochs': 5,
            'learning_rate': 0.0001,
            'beta_1': 0.5,
            'beta_2': 0.9,
            'decay': 0.000025,
            'd_step': 5,
            'wass_metric': False
        }
        if DATASET == 'mnist':
            x = x.reshape(x.shape[0], 1, 28, 28)
        client = clients.ClusterGANClient(x,
                                          y,
                                          config=config,
                                          client_id=CLIENT_ID)
    # TODO: elif args.alg == 'k-clustergan':
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=client)
