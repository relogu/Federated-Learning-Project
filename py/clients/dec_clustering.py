#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
import os
from typing import Dict, Union, Callable
from pathlib import Path

import numpy as np

from flwr.client import NumPyClient
from flwr.common import Parameters
from flwr.common.typing import Scalar

import py.metrics as my_metrics
from py.dec.util import (create_autoencoder, create_clustering_model, target_distribution)
from py.dumping.output import dump_result_dict

class DECClient(NumPyClient):
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
        # set
        self.n_clusters = config['n_clusters']
        # clustering model
        self.optimizer = config['optimizer']
        self.local_epochs = config['local_epochs']
        self.loss = config['loss']
        
        self.batch_size = config['batch_size']

        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
        
        # getting finetuned ae weights
        tmp = self.out_dir/'agg_weights_finetune_ae.npz'
        param: Parameters = np.load(
            tmp, allow_pickle=True)
        weights = param['arr_0']
        self.autoencoder, encoder, decoder = create_autoencoder(
            config, None
        )
        encoder.set_weights(weights)
        # initializing clustering model
        self.clustering_model = create_clustering_model(
            config['n_clusters'],
            encoder)
        # compiling the clustering model
        self.clustering_model.compile(
            optimizer=config['optimizer'],
            loss=config['loss'])
        # getting final clusters' centers
        tmp = self.out_dir/'agg_clusters_centers.npz'
        param: Parameters = np.load(
            tmp, allow_pickle=True)
        weights = np.array([param[p] for p in param])
        self.clustering_model.get_layer(
            name='clustering').set_weights(np.array([weights]))
        del tmp, param, encoder, decoder

        # default
        self.y_pred_filename = 'client_{}_dec_y_pred.npz'.format(self.client_id)
        self.p_filename = 'client_{}_dec_p.npz'.format(self.client_id)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        print('end init')
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        return self.clustering_model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        self.step = config['model']
        print("Client %s, Federated Round %d/%d, step: DEC clustering" % \
            (self.client_id, config['actual_round'], config['total_rounds']))
        # getting new weights
        self.clustering_model.set_weights(parameters)
        # fitting clustering model
        if config['update_interval']:
            print('Updating auxiliary distribution')
            self.update_interval = False
            q = self.clustering_model.predict(self.train['x'], verbose=0)
            # update the auxiliary target distribution p
            p = target_distribution(q)
            np.savez(self.out_dir/self.p_filename, *p)
        else:
            param: Parameters = np.load(
                self.out_dir/self.p_filename,
                allow_pickle=True)
            p = np.array([param[par] for par in param])
        for _ in range(int(self.local_epochs)):
            self.clustering_model.fit(
                x=self.train['x'],
                y=p,
                verbose=0)
        # returning the parameters necessary for FedAvg
        return self.clustering_model.get_weights(), len(self.train['x']), {}

    def _classes_evaluate(self, y_pred, config):
        metrics = {}
        # evaluating metrics
        acc = my_metrics.acc(self.test['y'], y_pred)
        nmi = my_metrics.nmi(self.test['y'], y_pred)
        ami = my_metrics.ami(self.test['y'], y_pred)
        ari = my_metrics.ari(self.test['y'], y_pred)
        ran = my_metrics.ran(self.test['y'], y_pred)
        homo = my_metrics.homo(self.test['y'], y_pred)
        if config['verbose']:
            print('Client %s, FedIter %d\n\tacc %.5f\n\tnmi '
                    '%.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' % \
                (self.client_id, config['actual_round'], acc, nmi, ami, ari, ran, homo))
        # dumping and retrieving the results
        metrics = {"accuracy": acc,
                    "normalized_mutual_info_score": nmi,
                    "adjusted_mutual_info_score": ami,
                    "adjusted_rand_score": ari,
                    "rand_score": ran,
                    "homogeneity_score": homo}
        return metrics

    def evaluate(self, parameters, config):
        metrics = {}
        self.clustering_model.set_weights(parameters)

        ## train loss computation
        q = self.clustering_model.predict(self.train['x'], verbose=0)
        train_y_pred = q.argmax(1)
        # update the auxiliary target distribution p
        p = target_distribution(q)
        # retrieving loss
        metrics['train_loss'] = self.clustering_model.evaluate(self.train['x'], p, verbose=0)

        ## eval loss computation
        q = self.clustering_model.predict(self.test['x'], verbose=0)
        eval_y_pred = q.argmax(1)
        # update the auxiliary target distribution p
        p = target_distribution(q)
        # retrieving loss
        metrics['eval_loss'] = self.clustering_model.evaluate(self.test['x'], p, verbose=0)

        ## eval clustering performance using some metrics
        # getting the train cycle accuracy
        y_ae_pred = self.clustering_model.predict(
            np.round(self.autoencoder(self.train['x'])),
            verbose=0).argmax(1)
        metrics['train_cycle_accuracy'] = my_metrics.acc(train_y_pred, y_ae_pred)
        # getting the eval cycle accuracy
        y_ae_pred = self.clustering_model.predict(
            np.round(self.autoencoder(self.test['x'])),
            verbose=0).argmax(1)
        metrics['eval_cycle_accuracy'] = my_metrics.acc(eval_y_pred, y_ae_pred)
        del y_ae_pred

        if (self.out_dir/self.y_pred_filename).exists():
            param: Parameters = np.load(
                self.out_dir/self.y_pred_filename,
                allow_pickle=True)
            eval_y_old = np.array([param[a] for a in param])
            metrics['tol'] = 1 - my_metrics.acc(eval_y_pred, eval_y_old)
        else:
            metrics['tol'] = 1.0
        np.savez(self.out_dir/self.y_pred_filename,
                 *eval_y_pred)
        metrics.update(self._classes_evaluate(eval_y_pred, config))
        metrics['client'] = self.client_id
        metrics['round'] = config['actual_round']
        if config['dump_metrics']:
            dump_result_dict('client_'+str(self.client_id),
                             metrics,
                             path_to_out=self.out_dir)
        return (metrics['eval_loss'], len(self.test['x']), metrics)
