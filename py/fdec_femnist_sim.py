#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import flwr as fl
import numpy as np
import os
import pathlib
import argparse
import json

from py.dec.util import create_dec_sae

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal

import py.metrics as my_metrics
from clients import AutoencoderClient, DECClient, KMeansClient
from strategies import SaveModelStrategy, KMeansStrategy, DECModelStrategy

def get_parser():
    # TODO: descriptor
    parser = argparse.ArgumentParser(
        description="FLOWER experiment for simulating the FEMNIST training")
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        default=1000,
                        action='store',
                        help='number of total clients in the FL setting')
    parser.add_argument("--n_clusters",
                        dest="n_clusters",
                        default=10,
                        type=int,
                        help="Define the number of clusters to identify")
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=False,
                        type=int,
                        default=50000,
                        action='store',
                        help='number of epochs for the autoencoder pre-training')
    parser.add_argument('--cl_epochs',
                        dest='cl_epochs',
                        required=False,
                        type=int,
                        default=10000,
                        action='store',
                        help='number of epochs for the clustering step')
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=140,
                        action='store',
                        help='set the update interval for the clusters distribution')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('--dump_metrics',
                        dest='dump_metrics',
                        required=False,
                        action='store_true',
                        help='Flag for dumping metrics during training and evaluation')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Flag for verbosity')
    parser.add_argument("--out_fol",
                        dest="out_folder",
                        type=type(str('')),
                        help="Folder to output images")
    parser.add_argument("--in_fol",
                        dest="in_folder",
                        type=type(str('')),
                        help="Folder to input images")
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
    # Define input folder
    if args.in_folder is None:
        path_to_in = pathlib.Path(__file__).parent.parent.absolute()/'input'
    else:
        path_to_in = pathlib.Path(args.in_folder)
    print('input folder {}'.format(path_to_in))
    
    # This might be changed
    client_resources = {'num_cpus': 1}
    # (optional) Specify ray config, for sure it is to be changed
    ray_config = {'include_dashboard': False}

    # Prepare FEMNIST dataset
    n_clients = len(list((path_to_in/'train').glob('*.json')))
    clients_ids = [a.parts[-1].split('.')[0] for a in list((path_to_in/'train').glob('*.json'))]
    idx = np.random.choice(range(n_clients), args.n_clients)
    n_clients = args.n_clients
    clients_ids = [clients_ids[i] for i in idx]
    print('Training for {} clients'.format(len(clients_ids)))
    ## PRETRAIN AE
    # Define AE configuration
    def lr_scheduler_fn(actual_round: int):
        # lr is divided by 10 every 20000 rounds
        return 0.1 - actual_round * (0.01-0.1) / 20000
    config = {
        'training_type': 'pretrain',
        'train_metrics': [my_metrics.rounded_accuracy, 'accuracy'],
        'batch_size': 64,
        'local_epochs': 1,
        'optimizer_lr_fn': lr_scheduler_fn,
        'create_ae_fn': create_dec_sae,
        'config_ae_args': {
          'dims': [784,
                 500,
                 500,
                 2000,
                 10],
          'init': RandomNormal(mean=0.0,
                             stddev=0.01),
          'dropout_rate': 0.2,
        },
        'loss': 'mse',
        'binary': False,
        'tied': True,
    }
    # Define get dataset fn for AE
    common_rand_state = np.random.randint(10000)
    def get_ae_dataset_fn(cid: str):
        # Must return (train, test) given the client_id as str
        # starting from parameters light in memory
        with open(path_to_in/'train'/str(cid+'.json'), 'r') as file:
            f = json.load(file)
            x_train = np.array(f['x'])
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[2]))
        with open(path_to_in/'test'/str(cid+'.json'), 'r') as file:
            f = json.load(file)
            x_test = np.array(f['x'])
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
        return x_train, x_test
    # Define the client fn
    def pae_client_fn(cid: str):
        # Create a single client instance from client id
        return AutoencoderClient(client_id=cid,
                                 config=config,
                                 get_data_fn=get_ae_dataset_fn,
                                 output_folder=path_to_out)
    # Define on_fit_config_fn
    pretrain_rounds = args.ae_epochs
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
        return {'dump_metrics': args.dump_metrics,
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
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=pae_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=clients_ids,
                                   client_resources=client_resources,
                                   num_rounds = pretrain_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
    ## FINETUNE AE
    # Refine AE configuration
    config['training_type'] = 'finetune'
    config['dropout'] = 0.0
    # Define the client fn
    def fae_client_fn(cid: str):
        # Create a single client instance from client id
        return AutoencoderClient(client_id=cid,
                                 config=config,
                                 get_data_fn=get_ae_dataset_fn,
                                 output_folder=path_to_out)
    # Define on_fit_config_fn
    finetune_rounds = 2*pretrain_rounds
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
        return {'dump_metrics': args.dump_metrics,
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
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=fae_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=clients_ids,
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
        with open(path_to_in/'train'/str(cid+'.json'), 'r') as file:
            f = json.load(file)
            x_train = np.array(f['x'])
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[2]))
            y_train = np.array(f['y'])
            y_train = y_train.reshape((y_train.shape[0]))
        with open(path_to_in/'test'/str(cid+'.json'), 'r') as file:
            f = json.load(file)
            x_test = np.array(f['x'])
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
            y_test = np.array(f['y'])
            y_test = y_test.reshape((y_test.shape[0]))
        return ({'x': x_train,
                'y': y_train},
                {'x': x_test,
                'y': y_test})
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
        return {'dump_metrics': args.dump_metrics,
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': kmeans_rounds}
    # TODO: Configure the strategy parameters, 
    # all clients must be available
    current_strategy = KMeansStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_kmeans_fn,
        on_evaluate_config_fn=on_eval_config_kmeans_fn,
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=kmeans_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=clients_ids,
                                   client_resources=client_resources,
                                   num_rounds=kmeans_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
    ## DEC CLUSTERING
    # DEC dataset fn is the same as kmeans
    # Define DEC configuration
    config['optimizer'] = SGD(
        learning_rate=0.1,
        momentum=0.9)
    config['local_epochs'] = 1
    config['loss'] = 'kld'
    # Define the client fn
    def dec_client_fn(cid: str):
        # Create a single client instance from client id
        return DECClient(client_id=cid,
                         config=config,
                         get_data_fn=get_kmeans_dataset_fn,
                         output_folder=path_to_out)
    # Define on_fit_config_fn
    dec_rounds = args.cl_epochs
    def on_fit_config_dec_fn(train: bool, rnd: int):
        # Must have 'last', 'model' for server necessities; 
        # 'actual_round', 'total_rounds', 'update_interval', 
        # 'train_or_not_param' for client necessities
        return {'model': 'dec',
                'last': rnd==dec_rounds,
                'actual_round': rnd,
                'total_rounds': dec_rounds,
                'update_interval': (rnd%args.update_interval==1),
                'train': train}
    # Define on_evaluate_config_fn
    def on_eval_config_dec_fn(rnd: int):
        # Must have 'dump_metrics', 'verbose', 'actual_round'
        # for client necessities; 'n_clusters' for server necessities
        return {'dump_metrics': args.dump_metrics,
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': dec_rounds,
                'model': 'dec'}
    # TODO: Configure the strategy parameters
    current_strategy = DECModelStrategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_dec_fn,
        on_evaluate_config_fn=on_eval_config_dec_fn,
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
    )
    # Launch the simulation
    fl.simulation.start_simulation(client_fn=dec_client_fn,
                                   num_clients=args.n_clients,
                                   clients_ids=clients_ids,
                                   client_resources=client_resources,
                                   num_rounds=dec_rounds,
                                   strategy=current_strategy,
                                   ray_init_args=ray_config)
