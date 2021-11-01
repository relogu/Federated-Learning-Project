#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import flwr as fl
import numpy as np
import os
import sys
import pathlib
import argparse

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal

import py.dataset_util as data_util
import py.metrics as my_metrics
import py.losses as my_losses
from py.clients import AutoencoderClient, DECClient, KMeansClient
from py.strategies import SaveModelStrategy, KMeansStrategy

def get_parser():
    # TODO: descriptor
    parser = argparse.ArgumentParser(
        description="FLOWER experiment for simulating the FEMNIST training")
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
    parser.add_argument('--fill',
                        dest='fill',
                        required=False,
                        action='store_true',
                        help='Flag for fill NaNs in dataset')
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        default=2,
                        action='store',
                        help='number of total clients in the FL setting')
    parser.add_argument("--n_clusters",
                        dest="n_clusters",
                        default=10,
                        type=int,
                        help="Define the number of clusters to identify")
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Flag for verbosity')
    parser.add_argument("--out_fol",
                        dest="out_folder",
                        type=type(str('')),
                        help="Folder to output images")
    # TODO: add arguments
    return parser
    

# TODO: write description
if __name__ == "__main__":

    args = get_parser().parse_args()
    # Set the seed for reproducibility
    np.random.seed(args.seed)
    # Define output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    
    # This might be changed
    client_resources = {'num_cpus': 10}
    # (optional) Specify ray config, for sure it is to be changed
    ray_config = {'include_dashboard': False}

    # Prepare EURMDS dataset
    for g in args.groups:
        if g not in data_util.EUROMDS_GROUPS:
            print('One of the given groups is not allowed.\nAllowed groups: {}'.
                  format(data_util.EUROMDS_GROUPS))
            sys.exit()
    for c in args.ex_col:
        if c not in data_util.get_euromds_cols():
            print('One of the given columns is not allowed.\nAllowed columns: {}'.
                  format(data_util.get_euromds_cols()))
            sys.exit()
    # Get dataset slice chosen
    print("Filling? {}".format(args.fill))
    fill = 2044 if args.fill else 0
    x = data_util.get_euromds_dataset(groups=args.groups,
                                      exclude_cols=args.ex_col,
                                      accept_nan=fill,
                                      verbose=args.verbose)
    # Get the number of features selected
    n_features = len(x.columns)
    x = np.array(x)
    # Separate test and train sets
    initial_state = np.random.randint(10000)
    train_idx, test_idx = data_util.split_dataset(
        x=x,
        splits=5, # 80-20 for test-train
        shuffle=True,
        fold_n=0, # select first fold
        r_state=initial_state # set the seed
    )
    x_train, x_test = x[train_idx], x[test_idx]
    ## FREQUENCIES
    # Get frenquencies for building random flipping layer
    up_frequencies = np.array([np.array(np.count_nonzero(
        x_train[:, i])/x_train.shape[0]) for i in range(n_features)])
    np.savez(path_to_out/'agg_weights_up_frequencies.npz', up_frequencies)
    del x, x_train, x_test
    ## PRETRAIN AE
    # Define AE configuration
    config = {
        'training_type': 'pretrain',
        'train_metrics': [my_metrics.rounded_accuracy, 'accuracy'],
        'batch_size': 10,
        'local_epochs': 1,
        # ?? decay=float(9/((2/5)*int(config['ae_epochs']))))  # from DEC paper
        'optimizer': SGD(
            learning_rate=0.01,
            momentum=0.9),
        # to be properly defined
        'loss': my_losses.FocalLoss,#'mse',#tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),#'binary_crossentropy',
        'binary': False,
        'tied': True,
        'dims': [n_features,
                 int((2/3)*(n_features)),
                 int((2/3)*(n_features)),
                 int((2.5)*(n_features)),
                 args.n_clusters],  # DEC paper proportions
        'init': RandomNormal(
            mean=0.0,
            stddev=0.2), #stddev=0.01) # DEC paper, is better
        'dropout': 0.2,
        'ran_flip': 0.2,  
        'act': 'selu',
        'ortho': False,
        'u_norm': True,          
    }
    # Define get dataset fn for AE
    common_rand_state = np.random.randint(10000)
    def get_ae_dataset_fn(cid: str):
        # Must return (train, test) given the client_id as str
        # starting from parameters light in memory
        client_id = int(cid)
        x = np.array(data_util.get_euromds_dataset(
            groups=args.groups,
            exclude_cols=args.ex_col,
            accept_nan=fill))
        x_train, x_test = x[train_idx], x[test_idx]
        client_train, _ = data_util.split_dataset(
            x=x_train,
            splits=args.n_clients,
            shuffle=True,
            fold_n=client_id, # select the client
            r_state=common_rand_state # set common seed
        )
        client_test, _ = data_util.split_dataset(
            x=x_test,
            splits=args.n_clients,
            shuffle=True,
            fold_n=client_id, # select the client
            r_state=common_rand_state # set common seed
        )
        return x_train[client_train], x_test[client_test]
    # Define the client fn
    def pae_client_fn(cid: str):
        # Create a single client instance from client id
        return AutoencoderClient(client_id=cid,
                                 config=config,
                                 get_data_fn=get_ae_dataset_fn,
                                 output_folder=path_to_out)
    # Define on_fit_config_fn
    pretrain_rounds = 1 # TODO: get param from arguments
    def on_fit_config_pae_fn(rnd: int):
        # Must have 'last', 'model' for server necessities; 
        # 'actual_round', 'total_rounds' for client necessities
        return {'model': 'pretrain_ae',
                'last': rnd==pretrain_rounds,
                'actual_round': rnd,
                'total_rounds': pretrain_rounds}
    # Define on_evaluate_config_fn
    def on_eval_config_pae_fn(rnd: int):
        # Must have 'dump_metrics', 'filename', 'verbose', 'actual_round'
        # for client necessities
        return {'dump_metrics': True, # TODO: get param from arguments
                # to output metrics in client_*_pretrain_ae.dat
                'filename': '_pretrain_ae',
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': pretrain_rounds}
    # TODO: Configure the strategy parameters
    current_strategy = SaveModelStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_pae_fn,
        on_evaluate_config_fn=on_eval_config_pae_fn,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=pae_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=[str(c) for c in list(range(args.n_clients))],
                                   client_resources=client_resources,
                                   num_rounds = pretrain_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
    ## FINETUNE AE
    # Refine AE configuration
    config['training_type'] = 'finetune'
    config['dropout'] = 0.0
    config['ran_flip'] = 0.0
    # Define the client fn
    def fae_client_fn(cid: str):
        # Create a single client instance from client id
        return AutoencoderClient(client_id=cid,
                                 config=config,
                                 get_data_fn=get_ae_dataset_fn,
                                 output_folder=path_to_out)
    # Define on_fit_config_fn
    finetune_rounds = 2*pretrain_rounds # TODO: get param from arguments
    def on_fit_config_fae_fn(rnd: int):
        # Must have 'last', 'model' for server necessities; 
        # 'actual_round', 'total_rounds' for client necessities
        return {'model': 'finetune_ae',
                'last': rnd==finetune_rounds,
                'actual_round': rnd,
                'total_rounds': finetune_rounds}
    # Define on_evaluate_config_fn
    def on_eval_config_fae_fn(rnd: int):
        # Must have 'dump_metrics', 'filename', 'verbose', 'actual_round'
        # for client necessities
        return {'dump_metrics': True, # TODO: get param from arguments
                # to output metrics in client_*_finetune_ae.dat
                'filename': '_finetune_ae',
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': finetune_rounds}
    # TODO: Configure the strategy parameters
    current_strategy = SaveModelStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_fae_fn,
        on_evaluate_config_fn=on_eval_config_fae_fn,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=fae_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=[str(c) for c in list(range(args.n_clients))],
                                   client_resources=client_resources,
                                   num_rounds = finetune_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
    ## KMEANS
    # Define get dataset fn for kmeans (same for dec)
    def get_kmeans_dataset_fn(cid: str):
        # Must return (train, test) given the client_id as str
        # starting from parameters light in memory, both train
        # and test are dictionaries with 'x' and 'y' fields
        client_id = int(cid)
        x = np.array(data_util.get_euromds_dataset(
            groups=args.groups,
            exclude_cols=args.ex_col,
            accept_nan=fill))
        x_train, x_test = x[train_idx], x[test_idx]
        prob = data_util.get_euromds_dataset(groups=['HDP'])
        y = []
        for label, row in prob.iterrows():
            if np.sum(row) > 0:
                y.append(row.argmax())
            else:
                y.append(-1)
        y = np.array(y)
        del prob
        y_train, y_test = y[train_idx], y[test_idx]
        client_train, _ = data_util.split_dataset(
            x=x_train,
            splits=args.n_clients,
            shuffle=True,
            fold_n=client_id, # select the client
            r_state=common_rand_state # set common seed
        )
        client_test, _ = data_util.split_dataset(
            x=x_test,
            splits=args.n_clients,
            shuffle=True,
            fold_n=client_id, # select the client
            r_state=common_rand_state # set common seed
        )
        return ({'x': x_train[client_train],
                'y': y_train[client_train]},
                {'x': x_test[client_test],
                'y': y_test[client_test]})
    # Define k-means configuration
    config['k_means_init'] = 'k-means++'
    config['n_clusters'] = args.n_clusters
    config['kmeans_max_iter'] = 300
    config['kmeans_n_init'] = 25
    config['seed'] = np.random.randint(10000)
    # Define the client fn
    def kmeans_client_fn(cid: str):
        # Create a single client instance from client id
        return KMeansClient(client_id=cid,
                            config=config,
                            get_data_fn=get_kmeans_dataset_fn,
                            output_folder=path_to_out)
    # Define on_fit_config_fn
    kmeans_rounds = 1
    def on_fit_config_kmeans_fn(rnd: int):
        # Must have 'last', 'model', 'n_clusters' for server necessities; 
        # 'actual_round', 'total_rounds' for client necessities
        return {'model': 'kmeans',
                'last': rnd==kmeans_rounds,
                'actual_round': rnd,
                'total_rounds': kmeans_rounds,
                'n_clusters': args.n_clusters}
    # Define on_evaluate_config_fn
    def on_eval_config_kmeans_fn(rnd: int):
        # Must have 'dump_metrics', 'verbose', 'actual_round'
        # for client necessities
        return {'dump_metrics': True, # TODO: get param from arguments
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': kmeans_rounds}
    # TODO: Configure the strategy parameters, 
    # all clients must be available
    current_strategy = KMeansStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_kmeans_fn,
        on_evaluate_config_fn=on_eval_config_kmeans_fn,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=kmeans_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=[str(c) for c in list(range(args.n_clients))],
                                   client_resources=client_resources,
                                   num_rounds=kmeans_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
    ## DEC CLUSTERING
    # # Define dec dataset fn, same as kmeans
    # def get_dec_dataset_fn(cid: str):
    #     return None
    # Define DEC configuration
    config['optimizer'] = SGD(
        learning_rate=0.001,
        momentum=0.9)
    config['local_epochs'] = 1
    config['loss'] = 'kld'
    config['batch_size'] = 10
    # Define the client fn
    def dec_client_fn(cid: str):
        # Create a single client instance from client id
        return DECClient(client_id=cid,
                         config=config,
                         get_data_fn=get_kmeans_dataset_fn,
                         output_folder=path_to_out)
    # Define on_fit_config_fn
    dec_rounds = 22 # TODO: get param from arguments
    def on_fit_config_dec_fn(rnd: int):
        # Must have 'last', 'model' for server necessities; 
        # 'actual_round', 'total_rounds', 'update_interval', 
        # 'train_or_not_param' for client necessities
        return {'model': 'dec',
                'last': rnd==dec_rounds,
                'actual_round': rnd,
                'total_rounds': dec_rounds,
                'update_interval': (rnd%20==1)}
    # Define on_evaluate_config_fn
    def on_eval_config_dec_fn(rnd: int):
        # Must have 'dump_metrics', 'verbose', 'actual_round'
        # for client necessities; 'n_clusters' for server necessities
        return {'dump_metrics': True, # TODO: get param from arguments
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': dec_rounds,
                'model': 'dec'}
    # TODO: Configure the strategy parameters
    current_strategy = SaveModelStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_dec_fn,
        on_evaluate_config_fn=on_eval_config_dec_fn,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=dec_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=[str(c) for c in list(range(args.n_clients))],
                                   client_resources=client_resources,
                                   num_rounds=dec_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
