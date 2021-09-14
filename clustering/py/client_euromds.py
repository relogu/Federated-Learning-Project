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

import py.my_clients as clients
import py.metrics as my_metrics
import py.dataset_util as data_util

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
    description = 'TODO'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--client_id',
                        dest='client_id',
                        required=True,
                        type=int,
                        action='store',
                        help='client identifier')
    parser.add_argument('--alg',
                        dest='alg',
                        required=True,
                        type=type(''),
                        default='k-means',
                        choices=['k-means', 'udec',
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
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        default=2,
                        choices=[2, 3, 4, 5, 6, 7, 8],
                        action='store',
                        help='number of total clients in the FL setting')
    parser.add_argument('--fold_n',
                        dest='fold_n',
                        required=True,
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
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('--lda',
                        dest='lda',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='wheater to apply LDA partitioning to the entire dataset')
    parser.add_argument('--groups',
                        dest='groups',
                        required=True,
                        action='append',
                        help='which groups of variables to use for EUROMDS dataset')
    parser.add_argument('--ex_col',
                        dest='ex_col',
                        required=True,
                        action='append',
                        help='which columns to exclude for EUROMDS dataset')
    parser.add_argument('--out_fol',
                        dest='out_fol',
                        required=False,
                        type=type(''),
                        default=None,
                        action='store',
                        help='select the output folder')
    parser.add_argument("--hardware_acc", dest="cuda_flag", action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("--binary", dest="binary", action='store_true',
                        help="Flag for using binary neurons in the network")
    parser.add_argument('--tied',
                        dest='tied',
                        required=False,
                        action='store_true',
                        help='Flag for using tied layers in autoencoder')
    parser.add_argument('--plotting',
                        dest='plotting',
                        required=False,
                        action='store_true',
                        help='Flag for plotting confusion matrix')
    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=0.01,
                        required=False,
                        action='store',
                        help='Flag for dropout layer in autoencoder')
    parser.add_argument('--ran_flip',
                        dest='ran_flip',
                        type=float,
                        default=0.05,
                        required=False,
                        action='store',
                        help='Flag for RandomFlipping layer in autoencoder')
    parser.add_argument('--ortho',
                        dest='ortho',
                        required=False,
                        action='store_true',
                        help='Flag for orthogonality regularizer in autoencoder (tied only)')
    parser.add_argument('--u_norm',
                        dest='u_norm',
                        required=False,
                        action='store_true',
                        help='Flag for unit norm constraint in autoencoder (tied only)')
    parser.add_argument('--cl_lr',
                        dest='cl_lr',
                        required=False,
                        type=float,
                        default=0.1,
                        action='store',
                        help='clustering model learning rate')
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=100,
                        action='store',
                        help='set the update interval for the clusters distribution')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Flag for verbosity')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":

    # parsing arguments
    args = parse_args()
    # disable possible gpu devices (add hard acc, selection)
    if not args.cuda_flag:
        print('No CUDA')
        tf.config.set_visible_devices([], 'GPU')

    # managing parameters
    CLIENT_ID = args.client_id
    N_CLIENTS = args.n_clients
    N_CLUSTERS = args.n_clusters
    SEED = args.seed
    USE_LDA = args.lda
    SERVER = args.server
    FOLD_N = args.fold_n
    SHUFFLE = args.shuffle
    print("Server address: "+SERVER)

    # initializing common configuration dict
    config = {
        'batch_size': 32,
        'splits': 5,
        'fold_n': FOLD_N,
        'n_clusters': N_CLUSTERS,
        'shuffle': SHUFFLE,
        'kmeans_local_epochs': 300,
        'kmeans_n_init': 25,
        'ae_local_epochs': 100,
        'ae_lr': 0.1,
        'ae_momentum': 0.9,
        'cl_lr': args.cl_lr,
        'cl_momentum': 0.9,
        'cl_local_epochs': 5,
        'update_interval': args.update_interval,
        'ae_loss': 'binary_crossentropy',#'mse',
        'cl_loss': 'kld',
        'seed': args.seed,
        'binary': args.binary,
        'plotting': False,
        'verbose': args.verbose}

    # preparing dataset
    for g in args.groups:
        if g not in data_util.EUROMDS_GROUPS:
            print('One of the given groups is not allowed.\nAllowed groups: {}'.\
                format(data_util.EUROMDS_GROUPS))
            sys.exit()
    for c in args.ex_col:
        if c not in data_util.get_euromds_cols():
            print('One of the given columns is not allowed.\nAllowed columns: {}'.\
                format(data_util.get_euromds_cols()))
            sys.exit()
    # getting the entire dataset
    x = data_util.get_euromds_dataset(groups=args.groups, exclude_cols=args.ex_col)
    # getting labels from HDP
    prob = data_util.get_euromds_dataset(groups=['HDP'])
    y = []
    for label, row in prob.iterrows():
        if np.sum(row) > 0:
            y.append(row.argmax())
        else:
            y.append(-1)
    del prob
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
    # # inferring ground truth labels usign HDBSCAN alg
    # y_h = hdbscan.HDBSCAN(
    #     min_samples=5,
    #     min_cluster_size=25,
    # ).fit_predict(x)
    # getting the client's dataset (partition)
    interval = int(len(x)/N_CLIENTS)
    start = int(interval*CLIENT_ID)
    end = int(interval*(CLIENT_ID+1)) if CLIENT_ID < N_CLIENTS-1 else len(x)
    x = np.array(x[start:end])
    y = y[start:end]
    outcomes = outcomes[start:end].reindex()
    outcomes = np.array(outcomes[['outcome_3', 'outcome_2']])
    ids = np.array(ids[start:end])
    # setting the autoencoder layers
    dims = [x.shape[-1],
            int((2/3)*(n_features)),
            int((2/3)*(n_features)),
            int((2.5)*(n_features)),
            N_CLUSTERS]
    init = VarianceScaling(scale=1. / 3.,
                           mode='fan_in',
                           distribution='uniform')

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

    config['ae_tied'] = args.tied
    config['ae_dims'] = dims
    config['ae_init'] = init
    config['ae_dropout_rate'] = args.dropout
    config['ae_flip_rate'] = args.ran_flip
    config['ae_ortho'] = args.ortho
    config['ae_u_norm'] = args.u_norm

    # algorithm choice
    if args.alg == 'k-means':
        config['kmeans_local_epochs'] = 1
        client = clients.SimpleKMeansClient(x=x,
                                            y=y,
                                            ids=ids,
                                            outcomes=outcomes,
                                            client_id=CLIENT_ID,
                                            seed=SEED,
                                            config=config,
                                            output_folder=args.out_fol)
    elif args.alg == 'udec' or args.alg == 'k_fed-ae_clust':
        client = clients.KMeansEmbedClusteringClient(x=x,
                                                     y=y,
                                                     outcomes=outcomes,
                                                     ids=ids,
                                                     client_id=CLIENT_ID,
                                                     config=config,
                                                     output_folder=args.out_fol)
    elif args.alg == 'clustergan':

        config = {
            'x_shape': x.shape[-1],
            'batch_size': 32,
            'splits': 5,
            'fold_n': FOLD_N,
            'n_clusters': N_CLUSTERS,
            'shuffle': SHUFFLE,
            'latent_dim': int(3*N_CLUSTERS),
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
            'disc_dims': [int(x.shape[-1]), int(2*n_features), int(3*n_features), int(4*n_features)],
            'use_binary': args.binary
        }

        client = clients.ClusterGANClient(x=x,
                                          y=y,
                                          outcomes=outcomes,
                                          ids=ids,
                                          config=config,
                                          client_id=CLIENT_ID,
                                          output_folder=args.out_fol)
    # TODO: elif args.alg == 'k-clustergan':
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=client)
