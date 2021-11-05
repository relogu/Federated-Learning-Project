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
from py.dec.util import create_autoencoder
from py.dumping.output import dump_result_dict

class AutoencoderClient(NumPyClient):
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
        # pretrain or finetune
        self.training_type = config['training_type']
        # train metrics
        self.train_metrics = config['train_metrics']
        # optimizer
        self.optimizer = config['optimizer']
        # loss
        self.loss = config['loss']
        self.batch_size = config['batch_size']
        self.local_epochs = config['local_epochs']
        # network arch
        self.net_arch = config
        self.binary = config['binary']
        self.tied = config['tied']
        self.dims = config['dims']
        self.init = config['init']
        self.dropout = config['dropout']
        self.act = config['act']
        self.ortho = config['ortho']
        self.u_norm = config['u_norm']
        self.ran_flip = config['ran_flip']
        
        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
        
        # TODO: solve frequencies problem
        # take the local frequencies?
        # make a global step with all the clients involved?
        # do something else?
        filename = self.out_dir / \
            'agg_weights_up_frequencies.npz'
        print('Found frequencies')
        param: Parameters = np.load(filename, allow_pickle=True)
        up_frequencies = param['arr_0']
        self.autoencoder, self.encoder, self.decoder = create_autoencoder(
            config, up_frequencies
        )
        self.autoencoder.compile(
            metrics=config['train_metrics'],
            optimizer=config['optimizer'],
            loss=config['loss']
        )
        # in case of finetuning traing type
        pretrained_weights = self.out_dir / \
            'agg_weights_pretrain_ae.npz'
        if pretrained_weights.exists() and (config['training_type'] == 'finetune'):
            print('Found pretrained weights')
            param: Parameters = np.load(pretrained_weights, allow_pickle=True)
            weights = param['arr_0']
            self.encoder.set_weights(weights)
        # general initializations
        self.last_histo = None
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_parameters(self):
        """Get the model weights by model object."""
        return self.encoder.get_weights()

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        self.step = config['model']
        print("Client %s, Federated Round %d/%d, step: %s" % \
            (self.client_id, config['actual_round'], config['total_rounds'], self.step))
        self.encoder.set_weights(parameters)
        # fitting the autoencoder
        self.autoencoder.fit(x=self.train,
                             y=self.train,
                             batch_size=self.batch_size,
                             epochs=self.local_epochs,
                             verbose=0)
        # returning the parameters necessary for FedAvg
        return self.encoder.get_weights(), len(self.train), {}

    def evaluate(self, parameters, config):
        metrics = {}
        self.encoder.set_weights(parameters)
        # training metrics
        loss, r_accuracy, accuracy = self.autoencoder.evaluate(
            self.train,
            self.train,
            verbose=0)
        metrics["train_loss"] = loss
        metrics["train_accuracy"] = accuracy
        metrics["train_r_accuracy"] = r_accuracy
        # evaluation
        loss, r_accuracy, accuracy = self.autoencoder.evaluate(
            self.test,
            self.test,
            verbose=0)
        metrics["eval_loss"] = loss
        metrics["eval_accuracy"] = accuracy
        metrics["eval_r_accuracy"] = r_accuracy
        # other quantities
        metrics['client'] = self.client_id
        metrics['round'] = config['actual_round']
        if config['dump_metrics']:
            dump_result_dict('client_'+str(self.client_id)+config['filename'],
                             metrics,
                             path_to_out=self.out_dir)
        if config['verbose']:
            print('Client %s, FedIter %d\n\tae_loss %.5f\n\taccuracy %.5f\n\tr_accuracy %.5f' % \
                (self.client_id, config['actual_round'], loss, metrics["eval_accuracy"], metrics["eval_r_accuracy"]))
        return (loss, len(self.test), {"r_accuracy": metrics["eval_r_accuracy"],
                                         "accuracy": metrics["eval_accuracy"]})
