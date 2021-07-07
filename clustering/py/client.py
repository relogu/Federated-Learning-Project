#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import argparse
import math
import os
import pathlib
import random
import sys
from argparse import RawTextHelpFormatter

import flwr as fl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import umap
import hdbscan
from flwr.dataset.utils.common import create_lda_partitions, create_partitions
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD

matplotlib.use('Agg')
path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(path.parent.parent))

import py.my_clients as clients
import py.metrics as my_metrics
import py.dataset_util as data_util
import clustering.py.common_fn as my_fn

# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"
# for limiting the cpu cores to use
torch.set_num_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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
                        choices=['blobs', 'moons', 'mnist', 'EUROMDS'],
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
    parser.add_argument('--dim_red',
                        dest='dim_red',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='set True if You want to do the dimensionality reduction')
    parser.add_argument('--n_samples',
                        dest='n_samples',
                        required=False,
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
    parser.add_argument('--groups',
                        dest='groups',
                        required=False,
                        type=int,
                        choices=[1,2,3,4,5,6,7],
                        default=7,
                        action='store',
                        help='how many groups of variables to use for EUROMDS dataset')
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

    # initializing common configuration dict
    config = {
        'batch_size': 32,
        'n_clusters': N_CLUSTERS,
        'kmeans_local_epochs': 300,
        'kmeans_n_init': 25,
        'ae_local_epochs': 5,
        'ae_lr': 0.01,
        'ae_momentum': 0.9,
        'cl_lr': 0.01,
        'cl_momentum': 0.9,
        'cl_local_epochs': 5,
        'update_interval': 55,
        'ae_loss': 'mse',
        'cl_loss': 'kld'}
    
    outcomes = None

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
        x, y = Y[CLIENT_ID]  # .copy()
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], int(4*n_features), N_CLUSTERS]
        del X, Y
    elif DATASET == 'moons':
        x_tot, y_tot = data_util.build_dataset(
            N_CLIENTS, N_SAMPLES, R_NOISE, SEED)
        x_train, y_train = data_util.get_client_dataset(
            args.client_id, N_CLIENTS, x_tot, y_tot)
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], 8, 8, 32, N_CLUSTERS]
        del x_tot, y_tot
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
        x, y = Y[CLIENT_ID].copy()
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 255.)
        # dimensions of the autoencoder dense layers
        dims = [x.shape[-1], 500, 500, 2000, N_CLUSTERS]
        del X, Y
    elif DATASET == 'EUROMDS':
        groups = ['Genetics', 'CNA', 'Demographics', 'Clinical', 'GeneGene', 'CytoCyto', 'GeneCyto']
        # getting the entire dataset
        x = data_util.get_euromds_dataset(groups=groups[:args.groups])
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
        # getting IDs
        ids = data_util.get_euromds_ids()
        n_features = len(x.columns)
        # dimensionality reduction through UMAP alg
        if args.dim_red:
            n_features = 2
            x = umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=n_features,
                random_state=42,
            ).fit_transform(x)
        # inferring ground truth labels usign HDBSCAN alg
        y_h = hdbscan.HDBSCAN(
            min_samples=5,
            min_cluster_size=25,
        ).fit_predict(x)
        # getting the client's dataset (partition)
        interval = int(len(x)/N_CLIENTS)
        start = int(interval*CLIENT_ID)
        end = int(interval*(CLIENT_ID+1)) if CLIENT_ID < N_CLIENTS-1 else len(x)
        x = np.array(x[start:end])
        y = y[start:end]
        outcomes = outcomes[start:end].reindex()
        ids = np.array(ids[start:end])
        # setting the autoencoder layers
        dims = [x.shape[-1], int(2*n_features), int(4*n_features), N_CLUSTERS]

        '''
        TODO: compatibility with create_partitions methods
        X = (x, y)
        if USE_LDA:
            Y, _ = create_lda_partitions(dataset=X, num_partitions=N_CLIENTS)
        else:
            Y = create_partitions(unpartitioned_dataset=X,
                                  iid_fraction=0.4,
                                  num_partitions=N_CLIENTS)
        x, y = Y[CLIENT_ID]
        del X, Y
        '''

    config['ae_dims'] = dims

    '''
    # kmeans baseline
    if y is not None:
        kmeans = KMeans(n_clusters=N_CLUSTERS,
                        verbose=0,
                        random_state=SEED)
        y_pred_kmeans = kmeans.fit_predict(x)
        print('Client %d, initial accuracy of k-Means: %.5f' %
              (CLIENT_ID, my_metrics.acc(y, y_pred_kmeans)))
        print('Client %d, initial accuracy of HDBSCAN: %.5f' %
              (CLIENT_ID, my_metrics.acc(y, y_h)))
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
        config['kmeans_local_epochs'] = 1
        client = clients.SimpleKMeansClient(x=x,
                                            y=y,
                                            seed=SEED,
                                            config=config)
    elif args.alg == 'k_fed-ae_clust':
        client = clients.KFEDClusteringClient(x=x,
                                              y=y,
                                              client_id=CLIENT_ID,
                                              config=config)
    elif args.alg == 'k-ae_clust':
        client = clients.KMeansEmbedClusteringClient(x=x,
                                                     y=y,
                                                     outcomes=outcomes,
                                                     client_id=CLIENT_ID,
                                                     config=config)
    elif args.alg == 'clustergan':
        config = {
            'x_shape': x.shape[-1],
            'batch_size': 32,
            'splits': 5,
            'fold_n': 0,
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
            'wass_metric': False,
            'save_images': False,
            'conv_net': False,
            'gen_dims': [int(4*n_features), int(3*n_features), int(2*n_features), x.shape[-1]],
            'enc_dims': [int(x.shape[-1]), int(4*n_features), int(3*n_features), int(2*n_features)],
            'disc_dims': [int(x.shape[-1]), int(2*n_features), int(3*n_features), int(4*n_features)]
        }
        if DATASET == 'mnist':
            x = x.reshape(x.shape[0], 1, 28, 28)
            config['x_shape'] = x[1:]
            config['save_images'] = True
            config['conv_net'] = True
        client = clients.ClusterGANClient(x=x,
                                          y=y,
                                          outcomes=np.array(outcomes[['outcome_3', 'outcome_2']]),
                                          ids=ids,
                                          config=config,
                                          client_id=CLIENT_ID)
    # TODO: elif args.alg == 'k-clustergan':
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=client)
