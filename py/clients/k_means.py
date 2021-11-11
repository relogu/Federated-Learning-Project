#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import os
import numpy as np

from flwr.client import NumPyClient
from flwr.common import Parameters
from flwr.common.typing import Scalar
from typing import Union, Callable, Dict
from pathlib import Path
from sklearn.cluster import KMeans
import py.metrics as my_metrics
from py.dumping.output import dump_result_dict
from tensorflow.keras.optimizers import SGD

class KMeansClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 client_id,  # id of the client
                 config: Dict = None,  # configuration dictionary
                 get_data_fn: Callable = None, # fn for getting dataset
                 output_folder: Union[Path, str] = None # output folder
                ):
        # get dataset
        self.client_id = client_id
        self.train, self.test = get_data_fn(client_id)
        self.config = config
        self.autoencoder, self.encoder, self.decoder = config['create_ae_fn'](
            **config['config_ae_args']
        )
        self.autoencoder.compile(
            metrics=config['train_metrics'],
            optimizer=SGD(),
            loss=config['loss']
        )

        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
            
        finetuned_weights = self.out_dir / \
            'agg_weights_finetune_ae.npz'
        if finetuned_weights.exists():
            print('Found finetuned weights')
            param: Parameters = np.load(finetuned_weights, allow_pickle=True)
            weights = param['arr_0']
            self.encoder.set_weights(weights)
        # general initializations
        self.kmeans = KMeans(init=config['k_means_init'],
                             n_clusters=config['n_clusters'],
                             max_iter=config['kmeans_max_iter'],
                             n_init=config['kmeans_n_init'],
                             random_state=config['seed'])
        self.clusters_centers = []
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_parameters(self):
        """Get the model weights by model object."""
        return self.clusters_centers

    def fit(self, parameters, config):
        """Perform the fit step after having assigned new weights."""
        print("Client %s, Federated Round %d/%d, step: k-means" % \
            (self.client_id, config['actual_round'], config['total_rounds']))
        # fitting clusters' centers using k-means
        self.kmeans.fit(self.encoder.predict(self.train['x']))
        self.clusters_centers = self.kmeans.cluster_centers_
        # returning the parameters necessary for k-FED
        return self.clusters_centers, len(self.train['x']), {}

    def evaluate(self, parameters, config):
        metrics = {}
        loss = 0.0
        # predicting labels
        self.kmeans = KMeans(init=np.array(parameters),
                             n_clusters=self.config['n_clusters'],
                             max_iter=self.config['kmeans_max_iter'],
                             n_init=1,
                             random_state=self.config['seed'])
        y_pred = self.kmeans.fit_predict(
            self.encoder.predict(self.test['x']))
        # evaluating metrics
        acc = my_metrics.acc(self.test['y'], y_pred)
        nmi = my_metrics.nmi(self.test['y'], y_pred)
        ami = my_metrics.ami(self.test['y'], y_pred)
        ari = my_metrics.ari(self.test['y'], y_pred)
        ran = my_metrics.ran(self.test['y'], y_pred)
        homo = my_metrics.homo(self.test['y'], y_pred)
        if config['verbose']:
            print('Client %s, FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' % \
                (self.client_id, config['actual_round'], acc, nmi, ami, ari, ran, homo))
        # dumping and retrieving the results
        metrics = {"accuracy": acc,
                   "normalized_mutual_info_score": nmi,
                   "adjusted_mutual_info_score": ami,
                   "adjusted_rand_score": ari,
                   "rand_score": ran,
                   "homogeneity_score": homo}
        if config['dump_metrics']:
            metrics['client'] = self.client_id
            metrics['round'] = config['actual_round']
            dump_result_dict('client_'+str(self.client_id)+'_k', metrics,
                                path_to_out=self.out_dir)
        # retrieving the results
        return (loss, len(self.test['x']), metrics)
