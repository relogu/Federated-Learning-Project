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
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from flwr.client import NumPyClient
from flwr.common.typing import Scalar
from typing import Union, Dict, Any, OrderedDict
from pathlib import Path

from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.utils import cluster_accuracy
from py.util import compute_centroid_np

def fit_kmeans_loop(
    dataloader: DataLoader = None, # dataloader for train
    device: str = 'cpu', # device to pass torch
    autoencoder: Module = None, # network
    kmeans: Any = None, # kmeans object
    scaler: Any = None, # scaler object
    use_emp_centroids: bool = False, # flag for using empirical centroids
):
    autoencoder.to(device)
    autoencoder.eval()
    data = []
    r_data = []
    features = []
    actual = []
    # form initial cluster centres
    for batch in dataloader:
        with torch.no_grad():
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
                actual.append(value.cpu())
                data.append(batch.cpu())
            batch = batch.to(device, non_blocking=True)
            r_batch = autoencoder(batch)
            f_batch = autoencoder.encoder(batch)
            r_data.append(r_batch.cpu())
            features.append(f_batch.cpu())
    predicted = kmeans.fit_predict(
        scaler.fit_transform(torch.cat(features).cpu().numpy()) if scaler is not None else torch.cat(features).numpy()
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
    return (
        predicted,
        torch.cat(actual).numpy(),
        centroids,
        torch.cat(data).numpy(),
        torch.cat(r_data).numpy(),
        torch.cat(features).numpy(),
        )

class KMeansClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 client_id: str,  # id of client
                 data_loader_config: Dict = None, # config of dataloader
                 net_config: Dict = None, # config of network
                 kmeans_config: Dict = None, # config of kmeans
                 scaler_config: Dict = None, # config of scaler
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
        self.ds_test = data_loader_config['get_test_fn'](client_id=eval(client_id))
        self.trainloader = data_loader_config['trainloader_fn'](self.ds_train)
        self.valloader = data_loader_config['valloader_fn'](self.ds_test)
        # get network
        self.noising = None
        if 'noising' in net_config.keys():
            self.noising = net_config['noising']
            net_config.pop('noising')
        if 'corruption' in net_config.keys():
            net_config.pop('corruption')
        self.autoencoder = StackedDenoisingAutoEncoder(**net_config)
        ae_params_filename = 'agg_weights_finetune_ae.npz' if self.noising is not None else 'agg_weights_pretrain_ae.npz'
        # with open(path_to_out/ae_params_filename, 'r') as file:
        ae_parameters = np.load(self.out_dir/ae_params_filename, allow_pickle=True)
        ae_parameters = [ae_parameters[a] for a in ae_parameters][0]
        self.set_parameters(ae_parameters)
        # kmeans initializations
        self.kmeans_config = kmeans_config
        self.use_emp_centroids = False
        if 'use_emp_centroids' in kmeans_config.keys():
            self.use_emp_centroids = self.kmeans_config['use_emp_centroids']
            kmeans_config.pop('use_emp_centroids')
        self.kmeans = KMeans(**self.kmeans_config)
        self.clusters_centers = []
        # get scaler
        self.scaler = None
        if scaler_config['name'] is not None:
            self.scaler = scaler_config['get_scaler_fn'](scaler_config['name']) if scaler_config['scaler'] != 'none' else None
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
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set model parameters from server: server must pass as initial 
        # parameter the final server parameters for the autoencoder
        # self.set_parameters(parameters)
        # fit kmeans
        predicted, actual, self.clusters_centers, data, r_data, features = fit_kmeans_loop(
            kmeans=self.kmeans,
            autoencoder=self.autoencoder,
            dataloader=self.trainloader,
            device=device,
            scaler=self.scaler,
            use_emp_centroids=self.use_emp_centroids,
        )
        couples = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            couples.append(self.clusters_centers[i])
            couples.append(np.sum(idx))
        # save clusters_centers
        np.savez(self.out_dir/'kmeans_cluster_centers_{}.npz'.format(self.client_id), *self.clusters_centers)
        # returning the parameters necessary for FedAvg
        # return self.clusters_centers, len(self.ds_train), 0.0
        return couples, len(self.ds_train), 0.0
    
    def evaluate(self, parameters, config):
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set clusters centers from server
        self.kmeans_config['init'] = np.array(parameters)
        self.kmeans_config['n_init'] = 1
        self.kmeans = KMeans(**self.kmeans_config)
        # fit kmeans
        predicted, actual, self.clusters_centers, data, r_data, features = fit_kmeans_loop(
            kmeans=self.kmeans,
            autoencoder=self.autoencoder,
            dataloader=self.trainloader,
            device=device,
            scaler=self.scaler,
        )
        # evaluate clustering
        cos_sil_score = 0
        eucl_sil_score = 0
        data_calinski_harabasz = 0
        feat_calinski_harabasz = 0
        
        if len(np.unique(predicted)) > 1:
            cos_sil_score = silhouette_score(
                X=data,
                labels=predicted,
                metric='cosine')
            eucl_sil_score = silhouette_score(
                X=features,
                labels=predicted,
                metric='euclidean')
            data_calinski_harabasz = calinski_harabasz_score(
                X=data,
                labels=predicted)
            feat_calinski_harabasz = calinski_harabasz_score(
                X=features,
                labels=predicted)
        reassignment, accuracy = cluster_accuracy(predicted, actual)
        # TODO: save metrics
        # save labels for predicted_previous of next step
        # with open(self.out_dir/'predicted_previous{}.npz'.format(self.client_id), 'w') as file:
        np.savez(self.out_dir/'predicted_previous{}.npz'.format(self.client_id), *predicted)
        # save kmeans predictions for final studies
        # with open(self.out_dir/'kmeans_predictions{}.npz'.format(self.client_id), 'w') as file:
        np.savez(self.out_dir/'kmeans_predictions{}.npz'.format(self.client_id), *predicted)
        # returning the parameters necessary for evaluation
        return float(accuracy), len(self.ds_test), {'cosine silhouette score': float(cos_sil_score),
                                                    'euclidean silhouette score': float(eucl_sil_score),
                                                    'data calinski harabasz score': float(data_calinski_harabasz),
                                                    'features calinski harabasz score': float(feat_calinski_harabasz)}
