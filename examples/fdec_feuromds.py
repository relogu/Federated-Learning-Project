#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import flwr as fl
from flwr.common import Parameters
import numpy as np
import os
import pathlib
from functools import partial

from torch.utils.data import DataLoader
from torch.nn import ReLU, Sigmoid
from torch.nn.modules.loss import MSELoss

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from py.clients.torch import AutoencoderClient, KMeansClient, DECClient
from py.strategies import (SaveModelStrategyFedAdam, SaveModelStrategyFedAvg,
                           SaveModelStrategyFedYogi, KMeansStrategy)
from py.datasets.feuromds import CachedfEUROMDS
from py.datasets.euromds import CachedEUROMDS
from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.dec import DEC
from py.dec.torch.layers import TruncatedGaussianNoise
from py.dec.torch.utils import get_main_loss, get_linears, get_ae_opt, get_scaler
from py.parsers.fdec_feuromds_parser import fdec_feuromds_parser as get_parser

# os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

# TODO: write description
if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()
    # Set the seed for reproducibility
    np.random.seed(args.seed)
    # Define output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'feuromds_unif_fedadam_maxmin'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    # Define data folder
    if args.data_folder is None:
        data_folder = pathlib.Path(__file__).parent.parent.absolute()/'data'/'euromds'
    else:
        data_folder = pathlib.Path(args.data_folder)
    print('Data folder {}'.format(data_folder))
    
    # set client resources for ray
    client_resources = {'num_cpus': args.n_cpus}
    # (optional) Specify ray config, for sure it is to be changed
    ray_config = {'include_dashboard': False}

    ## Prepare generalized AutoencoderClient for pretraining
    # Set dataloader configuration dict
    data_loader_config = {
        'get_train_fn': partial(
            CachedfEUROMDS,
            # exclude_cols=args.ex_col,
            # groups=args.groups,
            exclude_cols=['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
            groups=['Genetics', 'CNA'],
            fill_nans=args.fill_nans,
            get_hdp=True,
            get_outcomes=True,
            get_ids=True,
            n_clients=args.n_clients,
            balance=args.balance,
            seed=args.seed,
            verbose=False,
            path_to_data=data_folder),
        'get_test_fn': partial(
            CachedfEUROMDS,
            # exclude_cols=args.ex_col,
            # groups=args.groups,
            exclude_cols=['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
            groups=['Genetics', 'CNA'],
            fill_nans=args.fill_nans,
            get_hdp=True,
            get_outcomes=True,
            get_ids=True,
            n_clients=args.n_clients,
            balance=args.balance,
            seed=args.seed,
            verbose=False,
            path_to_data=data_folder),
        'trainloader_fn': partial(
            DataLoader,
            batch_size=args.ae_batch_size,
            shuffle=False),
        'valloader_fn': partial(
            DataLoader,
            batch_size=args.ae_batch_size,
            shuffle=False),
    }
    # Set loss configuration dict
    loss_config = {
        'eval_criterion': MSELoss,
        'get_loss_fn': get_main_loss,
        'params': {
            'name': 'mse'
            },
    }
    # Set network configuration dict
    # dataset = CachedEUROMDS(
    #     exclude_cols=args.ex_col,
    #     groups=args.groups,
    #     path_to_data=data_folder,
    #     fill_nans=args.fill_nans,
    #     get_hdp=False,
    #     verbose=True,
    # )
    # TODO: correct issue, is setting 1047
    n_features = 54#dataset.n_features
    # print("Number of features is: {}".format(n_features))
    # print("Features are: {}".format(dataset.columns_names))
    # del dataset
    net_config = {
        'noising': None if args.noising == 0 else TruncatedGaussianNoise(
            shape=n_features,
            stddev=args.noising,
            rate=1.0,
            ),
        'corruption': args.corruption,
        'dimensions': get_linears(args.linears, n_features, args.hidden_dimensions),
        'activation': ReLU(),
        'final_activation': ReLU(),
        'dropout': args.hidden_dropout,
        'is_tied': True,
    }
    # Set optimizer configuration dict
    ae_opt_config = {
        'optimizer_fn': get_ae_opt,
        'name': args.ae_opt,
        'dataset': 'euromds',
        'lr': args.ae_lr,
        'linears': args.linears,
    }
    if args.ae_opt == 'sgd':
        main_strategy = partial(SaveModelStrategyFedAvg)
    elif args.ae_opt == 'adam':
        main_strategy = partial(SaveModelStrategyFedAdam)
    elif args.ae_opt == 'yogi':
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
                'last': rnd==args.ae_epochs,
                'actual_round': rnd,
                'total_rounds': args.ae_epochs,
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
                'total_rounds': args.ae_epochs}
    # Mandatory set parameters
    autoencoder = StackedDenoisingAutoEncoder(
        get_linears(args.linears, n_features, args.hidden_dimensions),
        activation=ReLU(),
        final_activation=ReLU(),
        dropout=0.0,
        is_tied=True,
    )
    ae_parameters = [val.cpu().numpy() for _, val in autoencoder.state_dict().items()]
    # ae_parameters = np.load(
    #     pathlib.Path(__file__).parent.parent.absolute()/'input_weights'/'pretrain_ae.npz',
    #     allow_pickle=True)
    # ae_parameters = [ae_parameters[a] for a in ae_parameters][0]
    initial_param = fl.common.weights_to_parameters(ae_parameters)
    # Configure the strategy
    current_strategy = main_strategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_pae_fn,
        on_evaluate_config_fn=on_eval_config_pae_fn,
        min_available_clients=args.n_clients,
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
        initial_parameters=initial_param,
    )
    # Launch the simulation
    fl.simulation.start_simulation(
        client_fn=pae_client_fn,
        num_clients=args.n_clients,
        client_resources=client_resources,
        num_rounds=args.ae_epochs,
        strategy=current_strategy,
        ray_init_args=ray_config)
    
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
                    'last': rnd==args.ae_epochs,
                    'actual_round': rnd,
                    'total_rounds': args.ae_epochs,
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
                    'total_rounds': args.ae_epochs}
        ae_parameters = np.load(
            path_to_out/'agg_weights_pretrain_ae.npz',
            allow_pickle=True)
        ae_parameters = [ae_parameters[a] for a in ae_parameters][0]
        initial_param = fl.common.weights_to_parameters(ae_parameters)
        # Configure the strategy
        current_strategy = main_strategy(
            out_dir=path_to_out,
            on_fit_config_fn=on_fit_config_ftae_fn,
            on_evaluate_config_fn=on_eval_config_ftae_fn,
            min_available_clients=args.n_clients,
            min_fit_clients=args.n_clients,
            min_eval_clients=args.n_clients,
            initial_parameters=initial_param,
        )
        # Launch the simulation
        fl.simulation.start_simulation(
            client_fn=ftae_client_fn,
            num_clients=args.n_clients,
            client_resources=client_resources,
            num_rounds=args.ae_epochs,
            strategy=current_strategy,
            ray_init_args=ray_config)
    
    ## Prepare generalized KMeansClient for initializing clusters centers
    # Dataloader configuration dict is the same as before
    # Network configuration dict is the same as before
    # Set kmeans configuration dict
    kmeans_config = {
        'use_emp_centroids': args.use_emp_centroids,
        'n_clusters': args.n_clusters,
        'n_init': args.n_init,
        'random_state': args.seed,
        'max_iter': args.max_iter,
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
    filename = 'agg_weights_pretrain_ae.npz' if net_config['noising'] is None else 'agg_weights_finetune_ae.npz'
    ae_parameters = np.load(
        path_to_out/filename,
        allow_pickle=True)
    ae_parameters = [ae_parameters[a] for a in ae_parameters][0]
    initial_param = fl.common.weights_to_parameters(ae_parameters)
    # Configure the strategy
    current_strategy = KMeansStrategy(
        out_dir=path_to_out,
        method=args.kmeans_agg,
        on_fit_config_fn=on_fit_config_kmeans_fn,
        on_evaluate_config_fn=on_eval_config_kmeans_fn,
        min_available_clients=args.n_clients,
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
        initial_parameters=initial_param,
    )
    # Launch the simulation
    fl.simulation.start_simulation(
        client_fn=kmeans_client_fn,
        num_clients=args.n_clients,
        client_resources=client_resources,
        num_rounds=1,
        strategy=current_strategy,
        ray_init_args=ray_config)
    ## Prepare generalized DECClient for clustering step
    # Dataloader configuration dict changes only here:
    data_loader_config['trainloader_fn'] = partial(
        DataLoader,
        batch_size=args.dec_batch_size,
        shuffle=False)
    # Network configuration dict is the same as before
    # Set DEC configuration dict
    dec_config = {
        'n_clusters': args.n_clusters,
        'hidden_dimension': args.hidden_dimensions,
        'alpha': args.alpha,
    }
    # Set optimizer configuration dict
    dec_opt_config = {
        'optimizer_fn': get_ae_opt,
        'name': args.ae_opt,
        'dataset': 'euromds',
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
    def on_fit_config_dec_fn(rnd: int, train: bool = True):
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
    # Mandatory set initial parameters
    dec_model = DEC(
        cluster_number=dec_config['n_clusters'],
        hidden_dimension=dec_config['hidden_dimension'],
        encoder=autoencoder.encoder,
        alpha=dec_config['alpha'])
    dec_parameters = [val.cpu().numpy() for _, val in dec_model.state_dict().items()]
    # dec_parameters = np.load(
    #     pathlib.Path(__file__).parent.parent.absolute()/'input_weights'/'dec_model.npz',
    #     allow_pickle=True)
    # dec_parameters = [dec_parameters[a] for a in dec_parameters][0]
    initial_param = fl.common.weights_to_parameters(dec_parameters)
    # Configure the strategy
    current_strategy = main_strategy(
        out_dir=path_to_out,
        on_fit_config_fn=on_fit_config_dec_fn,
        on_evaluate_config_fn=on_eval_config_dec_fn,
        min_available_clients=args.n_clients,
        min_fit_clients=args.n_clients,
        min_eval_clients=args.n_clients,
        initial_parameters=initial_param,
    )
    # Launch the simulation
    fl.simulation.start_simulation(
        client_fn=dec_client_fn,
        num_clients=args.n_clients,
        num_rounds=args.dec_epochs,
        strategy=current_strategy,
        ray_init_args=ray_config)
