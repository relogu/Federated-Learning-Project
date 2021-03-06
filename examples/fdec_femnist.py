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
from functools import partial
import json

from torch.utils.data import DataLoader
from torch.nn import ReLU, Sigmoid
from torch.nn.modules.loss import MSELoss

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from py.clients.torch import AutoencoderClient, KMeansClient, DECClient
from py.strategies import SaveModelStrategyFedAdam, SaveModelStrategyFedAvg, SaveModelStrategyFedYogi, KMeansStrategy
from py.datasets.femnist import CachedFEMNIST
from py.datasets.bfemnist import CachedBFEMNIST
from py.dec.torch.layers import TruncatedGaussianNoise
from py.dec.torch.utils import get_main_loss, get_linears, get_opt, get_scaler
from py.parsers.fdec_femnist_parser import fdec_femnist_parser as get_parser

# Restrict the number of available GPUs to the first
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()
    # Set the seed for reproducibility
    np.random.seed(args.seed)
    # Define output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    # Dump current configuration
    with open(path_to_out/'config.json', 'w') as file:
        json.dump(vars(args), file)
    # Define data folder
    if args.data_folder is None:
        data_folder = pathlib.Path(__file__).parent.parent.absolute()/'data'/'femnist'
    else:
        data_folder = pathlib.Path(args.data_folder)
    print('Data folder {}'.format(data_folder))
    
    # FIXME: issue with ray simulation and gpu resources
    # Set client resources for ray
    client_resources = {'num_cpus': args.n_cpus}
    # (optional) Specify ray config, for sure it is to be changed
    ray_config = {'include_dashboard': False}
    # Set minimum number of available clients to run a federated round
    min_clients = args.min_clients

    ## Prepare generalized AutoencoderClient for pretraining
    # Set dataloader configuration dict
    data_loader_config = {
        'get_train_fn': partial(
            CachedBFEMNIST if args.binary else CachedFEMNIST,
            train=True,
            device=args.device,
            seed=args.seed,
            path_to_data=data_folder),
        'get_test_fn': partial(
            CachedBFEMNIST if args.binary else CachedFEMNIST,
            train=False,
            device=args.device,
            seed=args.seed,
            path_to_data=data_folder),
        'trainloader_fn': partial(
            DataLoader,
            batch_size=args.ae_batch_size,
            shuffle=True),
        'valloader_fn': partial(
            DataLoader,
            batch_size=args.ae_batch_size,
            shuffle=False),
    }
    # Set loss configuration dict
    # TODO: check for mod loss config
    loss_config = {
        'eval_criterion': MSELoss,
        'get_loss_fn': get_main_loss,
        'params': {
            'name': args.main_loss
            },
    }
    # Set network configuration dict
    net_config = {
        'noising': TruncatedGaussianNoise(
            shape=784,
            stddev=args.noising,
            rate=1.0,
            ),
        'corruption': args.corruption,
        'dimensions': get_linears(args.linears, 784, args.hidden_dimensions),
        'activation': ReLU() if args.activation == 'relu' else Sigmoid(),
        'final_activation': ReLU() if args.final_activation == 'relu' else Sigmoid(),
        'dropout': args.hidden_dropout,
        'is_tied': True,
    }
    # Set optimizer configuration dict
    ae_opt_config = {
        'optimizer_fn': get_opt,
        'dataset': 'bmnist' if args.binary else 'mnist',
        'linears': args.linears,
        'lr': args.ae_lr,
        'optimizer': args.optimizer
    }
    # TODO: here should be possible to set server params for optimizers
    if args.optimizer == 'sgd':
        main_strategy = partial(SaveModelStrategyFedAvg)
    elif args.optimizer == 'adam':
        main_strategy = partial(SaveModelStrategyFedAdam)
    elif args.optimizer == 'yogi':
        main_strategy = partial(SaveModelStrategyFedYogi)
    # Define the client fn to pass ray simulation
    def pae_client_fn(cid: int):
        # Create a single client instance from client id
        return AutoencoderClient(
            client_id=cid,
            data_loader_config=data_loader_config,
            loss_config=loss_config,
            net_config=net_config,
            opt_config=ae_opt_config,
            output_folder=path_to_out)
    # Define on_fit_config_fn
    def on_fit_config_pae_fn(rnd: int):
        # Must have 'last', 'model' for server necessities; 
        # 'actual_round', 'total_rounds' for client necessities
        return {'model': 'pretrain_ae',
                'last': rnd==args.pretrain_epochs,
                'actual_round': rnd,
                'total_rounds': args.pretrain_epochs,
                'n_epochs': args.n_local_epochs}
    # Define on_evaluate_config_fn
    def on_eval_config_pae_fn(rnd: int):
        # Must have 'dump_metrics', 'filename', 'verbose', 'actual_round'
        # for client necessities
        return {'dump_metrics': args.dump_metrics,
                # to output metrics in client_*_pretrain_ae.dat
                'filename': '_pretrain_ae',
                'verbose': args.verbose,
                'actual_round': rnd,
                'total_rounds': args.pretrain_epochs}
    # Configure the strategy
    current_strategy = main_strategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_pae_fn,
        on_evaluate_config_fn=on_eval_config_pae_fn,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_eval_clients=min_clients,
    )
    # Launch the simulation, try-except clause is to avoid warnings
    try:
        fl.simulation.start_simulation(
            client_fn=pae_client_fn,
            num_clients=args.n_clients,
            client_resources=client_resources,
            num_rounds=args.pretrain_epochs,
            strategy=current_strategy,
            ray_init_args=ray_config)
    except ImportError:
        print('TSAE federated pretraining failed! Ray may still be missing.')
    
    if args.noising > 0:
        ## Prepare generalized AutoencoderClient for finetuning
        # Dataloader configuration dict is the same as before
        # Loss configuration dict is the same as before
        # Network configuration dict changes here only:
        net_config['noising'] = 0.0
        # Optimizer configuration dict is the same as before
        # Define the client fn to pass ray simulation
        def ftae_client_fn(cid: int):
            # Create a single client instance from client id
            return AutoencoderClient(
                client_id=cid,
                data_loader_config=data_loader_config,
                loss_config=loss_config,
                net_config=net_config,
                opt_config=ae_opt_config,
                output_folder=path_to_out)
        # Define on_fit_config_fn
        def on_fit_config_ftae_fn(rnd: int):
            # Must have 'last', 'model' for server necessities; 
            # 'actual_round', 'total_rounds' for client necessities
            return {'model': 'finetune_ae',
                    'last': rnd==args.finetune_epochs,
                    'actual_round': rnd,
                    'total_rounds': args.finetune_epochs,
                    'n_epochs': args.n_local_epochs}
        # Define on_evaluate_config_fn
        def on_eval_config_ftae_fn(rnd: int):
            # Must have 'dump_metrics', 'filename', 'verbose', 'actual_round'
            # for client necessities
            return {'dump_metrics': args.dump_metrics,
                    # to output metrics in client_*_finetune_ae.dat
                    'filename': '_finetune_ae',
                    'verbose': args.verbose,
                    'actual_round': rnd,
                    'total_rounds': args.finetune_epochs}
        # Configure the strategy
        current_strategy = main_strategy(
            out_dir=path_to_out,
            on_fit_config_fn=on_fit_config_ftae_fn,
            on_evaluate_config_fn=on_eval_config_ftae_fn,
            min_available_clients=min_clients,
            min_fit_clients=min_clients,
            min_eval_clients=min_clients,
        )
        # Launch the simulation, try-except clause is to avoid warnings
        try:
            fl.simulation.start_simulation(
                client_fn=ftae_client_fn,
                num_clients=args.n_clients,
                client_resources=client_resources,
                num_rounds=args.finetune_epochs,
                strategy=current_strategy,
                ray_init_args=ray_config)
        except ImportError:
            print('TSAE federated finetuning failed! Ray may still be missing.')
    
    if args.train_dec:
        ## Prepare generalized KMeansClient for initializing clusters centers
        # Dataloader configuration dict is the same as before
        # Network configuration dict is the same as before
        # Set kmeans configuration dict
        kmeans_config = {
            'n_clusters': args.n_clusters,
            'n_init': args.n_init,
            'random_state': args.seed,
            'init': 'k-means++',
        }
        # Set scaler configuration dict
        scaler_config = {
            'get_scaler_fn': get_scaler,
            'name': args.scaler,
        }
        # Define the client fn to pass ray simulation
        def kmeans_client_fn(cid: int):
            # Create a single client instance from client id
            return KMeansClient(
                client_id=cid,
                data_loader_config=data_loader_config,
                net_config=net_config,
                kmeans_config=kmeans_config,
                scaler_config=scaler_config,
                output_folder=path_to_out)
        # Define on_fit_config_fn
        def on_fit_config_kmeans_fn(rnd: int):
            # Must have 'last', 'model', 'n_clusters' for server necessities; 
            # 'actual_round', 'total_rounds' for client necessities
            return {'model': 'kmeans',
                    'last': rnd==1,
                    'actual_round': rnd,
                    'total_rounds': 1,
                    'n_clusters': args.n_clusters}
        # Define on_evaluate_config_fn
        def on_eval_config_kmeans_fn(rnd: int):
            # Must have 'dump_metrics', 'verbose', 'actual_round'
            # for client necessities
            return {'dump_metrics': args.dump_metrics,
                    'verbose': args.verbose,
                    'actual_round': rnd,
                    'total_rounds': 1}
        # Get SDAE parameters
        ae_params_filename = 'agg_weights_finetune_ae.npz' if args.noising > 0 else 'agg_weights_pretrain_ae.npz'
        with open(path_to_out/ae_params_filename, 'r') as file:
            ae_parameters = np.load(file, allow_pickle=True)
        # Configure the strategy
        current_strategy = KMeansStrategy(
            out_dir=path_to_out,
            on_fit_config_fn=on_fit_config_kmeans_fn,
            on_evaluate_config_fn=on_eval_config_kmeans_fn,
            # TODO: check for memory issues, but
            # a fraction of all the available clients
            # might not be sufficient
            min_available_clients=min_clients,
            min_fit_clients=min_clients,
            min_eval_clients=min_clients,
            initial_parameters=ae_parameters,
        )
        # Launch the simulation, try-except clause is to avoid warnings
        try:
            fl.simulation.start_simulation(
                client_fn=kmeans_client_fn,
                num_clients=args.n_clients,
                client_resources=client_resources,
                num_rounds=1,
                strategy=current_strategy,
                ray_init_args=ray_config)
        except ImportError:
            print('Federated initialization of centroids failed! Ray may still be missing.')
        ## Prepare generalized DECClient for clustering step
        # Dataloader configuration dict changes only here:
        data_loader_config['trainloader_fn'] = partial(
            DataLoader,
            batch_size=args.dec_batch_size,
            shuffle=True)
        # Network configuration dict is the same as before
        # Set DEC configuration dict
        dec_config = {
            'n_clusters': args.n_clusters,
            'hidden_dimension': args.hidden_dimensions,
            'alpha': args.alpha,
        }
        # Set optimizer configuration dict
        dec_opt_config = {
            'optimizer_fn': get_opt,
            'dataset': 'bmnist' if args.binary else 'mnist',
            'linears': args.linears,
            'lr': args.dec_lr,
        }
        # Define the client fn to pass ray simulation
        def dec_client_fn(cid: str):
            # Create a single client instance from client id
            return DECClient(
                client_id=cid,
                data_loader_config=data_loader_config,
                net_config=net_config,
                dec_config=dec_config,
                opt_config=dec_opt_config,
                output_folder=path_to_out)
        # Define on_fit_config_fn
        def on_fit_config_dec_fn(train: bool, rnd: int):
            # Must have 'last', 'model' for server necessities; 
            # 'actual_round', 'total_rounds', 'update_interval', 
            # 'train_or_not_param' for client necessities
            return {'model': 'dec',
                    'last': rnd==args.dec_epochs,
                    'actual_round': rnd,
                    'total_rounds': args.dec_epochs,
                    'train': train,
                    'n_epochs': args.n_local_epochs}
        # Define on_evaluate_config_fn
        def on_eval_config_dec_fn(rnd: int):
            # Must have 'dump_metrics', 'verbose', 'actual_round'
            # for client necessities; 'n_clusters' for server necessities
            return {'dump_metrics': args.dump_metrics,
                    'verbose': args.verbose,
                    'actual_round': rnd,
                    'total_rounds': args.dec_epochs,
                    'model': 'dec'}
        # Configure the strategy
        current_strategy = main_strategy(
            out_dir=path_to_out,
            on_fit_config_fn=on_fit_config_dec_fn,
            on_evaluate_config_fn=on_eval_config_dec_fn,
            min_available_clients=min_clients,
            min_fit_clients=min_clients,
            min_eval_clients=min_clients,
        )
        # Launch the simulation, try-except clause is to avoid warnings
        try:
            fl.simulation.start_simulation(
                client_fn=dec_client_fn,
                num_clients=args.n_clients,
                num_rounds=args.dec_epochs,
                strategy=current_strategy,
                ray_init_args=ray_config)
        except ImportError:
            print('Federated DEC training failed! Ray may still be missing.')
