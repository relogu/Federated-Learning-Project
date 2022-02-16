#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import os

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from flwr.client import NumPyClient
from flwr.common.typing import Scalar
from typing import Union, Dict, Any, OrderedDict
from pathlib import Path

from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.util import compute_centroid_np

def fit_kmeans_loop(
    dataloader: DataLoader = None, # dataloader for train
    device: str = 'cpu', # device to pass torch
    autoencoder: Module = None, # network
    kmeans: Any = None, # kmeans object
    scaler: Any = None, # scaler object
    use_emp_centroids: bool = False, # flag for using empirical centroids
):
    features = []
    actual = []
    # form initial cluster centres
    for batch in dataloader:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        batch = batch.to(device, non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    predicted = kmeans.fit_predict(
        scaler.fit_transform(torch.cat(features).numpy()) if scaler is not None else torch.cat(features).numpy()
    )
    centroids = kmeans.cluster_centers_
    # if choosing empirical centroids or kmeans centroids
    if use_emp_centroids:
        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))
        centroids = np.array(emp_centroids)
    # return predicted labels and centroids
    return (predicted, centroids)

class KMeansClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 client_id: str,  # id of client
                 data_loader_config: Dict = None, # config of dataloader
                 net_config: Dict = None, # config of network
                 kmeans_config: Dict = None, # config of kmeans
                 scaler_config: Dict = None, # config of scaler
                 device: str = 'cpu', # device to pass torch
                 output_folder: Union[Path, str] = None # output folder
                 ):
        # set output folder        
        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
        self.client_id = client_id
        # get datasets
        self.ds_train = data_loader_config['get_train_fn'](client_id=eval(client_id))
        self.ds_test = data_loader_config['get_test_fn'](client_id=eavl(client_id))
        self.trainloader = data_loader_config['trainloader_fn'](self.ds_train)
        self.valloader = data_loader_config['valloader_fn'](self.ds_test)
        # get network
        if 'noising' in net_config.keys():
            net_config.pop('noising')
        if 'corruption' in net_config.keys():
            net_config.pop('corruption')
        self.autoencoder = StackedDenoisingAutoEncoder(**net_config)
        # kmeans initializations
        self.kmeans_config = kmeans_config
        self.use_emp_centroids = False
        if 'use_emp_centroids' in kmeans_config.keys():
            self.use_emp_centroids = self.kmeans_config['use_emp_centroids']
            kmeans_config.pop('use_emp_centroids')
        self.kmeans = KMeans(**self.kmeans_config)
        self.clusters_centers = []
        # get scaler
        self.scaler = scaler_config['get_scaler_fn'](scaler_config['name']) if scaler_config['scaler'] != 'none' else None
        # set device
        self.device = device
        # general initializations
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_parameters(self):
        """Get the model weights by model object."""
        return self.clusters_centers
    
    def set_parameters(self, parameters):
        """Set the model weights by parameters object."""
        params_dict = zip(self.autoencoder.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.autoencoder.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # set model parameters from server: server must pass as initial 
        # parameter the final server parameters for the autoencoder
        self.set_parameters(parameters)
        # fit kmeans
        predicted, self.clusters_centers = fit_kmeans_loop(
            kmeans=self.kmeans,
            autoencoder=self.autoencoder,
            dataloader=self.trainloader,
            device=self.device,
            scaler=self.scaler,
            use_emp_centroids=self.use_emp_centroids,
        )
        # save clusters_centers
        with open(self.out_dir/'kmeans_cluster_centers_{}.npz'.format(self.client_id), 'w') as file:
            np.savez(file, *self.clusters_centers)
        # returning the parameters necessary for FedAvg
        return {self.clusters_centers}, {len(self.ds_train)}, {}
    
    def evaluate(self, parameters, config):
        # set clusters centers from server
        self.kmeans_config['init'] = np.array(parameters)
        self.kmeans = KMeans(**self.kmeans_config)
        # fit kmeans
        predicted, self.clusters_centers = fit_kmeans_loop(
            kmeans=self.kmeans,
            autoencoder=self.autoencoder,
            dataloader=self.trainloader,
            device=self.device,
            scaler=self.scaler,
        )
        # TODO: evaluate clustering
        # save labels for predicted_previous of next step
        with open(self.out_dir/'predicted_previous{}.npz'.format(self.client_id), 'w') as file:
            np.savez(file, *predicted)
        with open(self.out_dir/'kmeans_predictions{}.npz'.format(self.client_id), 'w') as file:
            np.savez(file, *predicted)
        # returning the parameters necessary for evaluation
        return {}, {len(self.ds_test)}, {}
