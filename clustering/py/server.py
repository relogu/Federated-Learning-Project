#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import os
from typing import Callable, Dict, List, Optional, Tuple
import argparse
from argparse import RawTextHelpFormatter
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Weights
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans
from functools import partial
import pathlib
import sys
path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(path.parent.parent))
import clustering.py.common_fn as my_fn
import clustering.py.my_strategies as strategies
# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"


def parse_args():
    """Parse the arguments passed."""
    description = 'Server program for moons test FL network using flower.\n' + \
        'Give the number of federated rounds to pass to the strategy builder.\n' + \
        'Give the minimum number of clients to wait for a federated averaging step.\n' + \
        'Give optionally the complete address onto which instantiate the server.'

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--n_clients',
                        dest='clients',
                        required=True,
                        type=int,
                        default=2,
                        choices=[2, 3, 4, 5, 6, 7, 8],
                        action='store',
                        help='minimum number of active clients to perform an iteration step')
    parser.add_argument('--strategy',
                        dest='strategy',
                        required=False,
                        type=type(''),
                        default='fed_avg',
                        choices=['fed_avg', 'k-fed', 'fed_avg_k-means'],
                        action='store',
                        help='strategy for the server')
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=False,
                        type=int,
                        default=300,
                        action='store',
                        help='number of federated epoch to preform the autoencoder step')
    parser.add_argument('--kmeans_epochs',
                        dest='kmeans_epochs',
                        required=False,
                        type=int,
                        default=1,
                        action='store',
                        help='number of federated epoch to preform the k-means step')
    parser.add_argument('--cluster_epochs',
                        dest='cluster_epochs',
                        required=False,
                        type=int,
                        default=1000,
                        action='store',
                        help='number of federated epoch to preform the clustering step')
    parser.add_argument('--address',
                        dest='address',
                        required=False,
                        type=type(''),
                        default='[::]:51550',
                        action='store',
                        help='complete address to launch server, e.g. 127.0.0.1:8081')

    _args = parser.parse_args()

    return _args


# number of communities
K = 5
# random seed
SEED = 51550

# Start Flower server
if __name__ == "__main__":
    # parsing arguments
    args = parse_args()
    # instantiating the strategy
    if args.strategy == 'fed_avg':
        on_fit_conf = partial(my_fn.simple_clustering_on_fit_config,
                              ae_epochs=args.ae_epochs,
                              kmeans_epochs=args.kmeans_epochs,
                              cl_epochs=args.cluster_epochs)
        n_rounds = args.ae_epochs+args.kmeans_epochs+args.cluster_epochs
        strategy = strategies.FedAvg(
            min_available_clients=args.clients,
            min_fit_clients=args.clients,
            min_eval_clients=args.clients,
            on_fit_config_fn=on_fit_conf
        )
    elif args.strategy == 'fed_avg_k-means':
        on_fit_conf = partial(my_fn.simple_kmeans_on_fit_config,
                              kmeans_epochs=args.kmeans_epochs)
        n_rounds = args.kmeans_epochs
        strategy = strategies.FedAvg(
            min_available_clients=args.clients,
            min_fit_clients=args.clients,
            min_eval_clients=args.clients,
            on_fit_config_fn=on_fit_conf
        )
    elif args.strategy == 'k-fed':
        on_fit_conf = partial(my_fn.kfed_clustering_on_fit_config,
                              ae_epochs=args.ae_epochs,
                              n_clusters=10,
                              cl_epochs=args.cluster_epochs)
        n_rounds = 1+args.ae_epochs+args.cluster_epochs
        strategy = strategies.KFEDStrategy(
            min_available_clients=args.clients,
            min_fit_clients=args.clients,
            min_eval_clients=args.clients,
            on_fit_config_fn=on_fit_conf
        )
    # setting the server complete address
    SERVER = args.address
    # starting the server
    fl.server.start_server(SERVER,
                           config={"num_rounds": n_rounds},
                           strategy=strategy)
