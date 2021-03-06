#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import os

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from flwr.common.typing import Scalar
from typing import Union, Dict, Any, OrderedDict, Iterable
from pathlib import Path

from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.utils import get_ae_lr

def training_loop(
    n_epochs: int = 1, # number of epochs
    dataloader: DataLoader = None, # dataloader for train
    device: str = 'cpu', # device to pass torch
    optimizer: Any = None, # optimizer for train
    loss_fn: Iterable[Module] = None, # loss(es) for train
    autoencoder: Module = None, # network
    noising: Module = None, # input noising module
    corruption: float = 0.0, # input corruption percentage
):
    autoencoder.to(device)
    if noising is not None:
        noising.to(device)
    autoencoder.train()
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    
    for _ in range(n_epochs):
        ret_loss = 0.0
        for i, batch in enumerate(dataloader):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                batch = batch[0]
            batch = batch.to(device)
            
            if noising is not None:
                batch = noising(batch)
            if corruption > 0:
                batch = F.dropout(batch, corruption)
            output = autoencoder(batch)
            
            losses = [l_fn_i(output, batch) for l_fn_i in loss_functions]
            loss = sum(losses)/len(loss_fn)
            ret_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
        ret_loss = ret_loss / (i+1)
    return ret_loss

def evaluating_loop(
    valloader: DataLoader = None, # dataloader for eval
    criterion: Any = None, # loss criterion
    device: str = 'cpu', # device to pass torch
    autoencoder: Module = None, # network
):
    autoencoder.to(device)
    val_loss = 0.0
    criterion = criterion()
    autoencoder.eval()
    for i, val_batch in enumerate(valloader):
        with torch.no_grad():
            if (
                isinstance(val_batch, tuple) or isinstance(val_batch, list)
            ) and len(val_batch) in [1, 2]:
                val_batch = val_batch[0]
            val_batch = val_batch.to(device)
            validation_output = autoencoder(val_batch)
            loss = criterion(validation_output, val_batch)
            val_loss += loss.item()
    return (val_loss / (i+1))

class AutoencoderClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 client_id: str,  # id of client
                 data_loader_config: Dict = None, # config of dataloader
                 loss_config: Dict = None, # config of loss fn
                 net_config: Dict = None, # config of network
                 opt_config: Dict = None, # config of optimizer
                 output_folder: Union[Path, str] = None # output folder
                 ):
        # get datasets
        self.ds_train = data_loader_config['get_train_fn'](client_id=eval(client_id))
        self.ds_test = data_loader_config['get_test_fn'](client_id=eval(client_id))
        self.trainloader = data_loader_config['trainloader_fn'](self.ds_train)
        self.valloader = data_loader_config['valloader_fn'](self.ds_test)
        # get loss
        self.eval_criterion = loss_config['eval_criterion']
        self.loss_fn = [loss_config['get_loss_fn'](**loss_config['params'])]
        # get network
        self.noising = None
        if 'noising' in net_config.keys():
            self.noising = net_config['noising']
            net_config.pop('noising')
        self.corruption = None
        if 'corruption' in net_config.keys():
            self.corruption = net_config['corruption']
            net_config.pop('corruption')
        self.autoencoder = StackedDenoisingAutoEncoder(**net_config)
        # get optimizer
        self.optimizer = opt_config['optimizer_fn'](
            opt=opt_config['optimizer'],
            lr=opt_config['lr'] if opt_config['lr'] is not None else get_ae_lr(
                dataset=opt_config['dataset'],
                linears=opt_config['linears'],
                opt=opt_config['optimizer'],
                )
        )(self.autoencoder.parameters())
        # set output folder        
        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
        # general initializations
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def get_parameters(self):
        """Get the model weights by model object."""
        return [val.cpu().numpy() for _, val in self.autoencoder.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the model weights by parameters object."""
        params_dict = zip(self.autoencoder.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.autoencoder.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set model parameters from server
        self.set_parameters(parameters)
        # fit the model
        loss = training_loop(
            n_epochs=config['n_epochs'],
            dataloader=self.trainloader,
            device=device,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            autoencoder=self.autoencoder,
            noising=self.noising,
            corruption=self.corruption,
        )
        # returning the parameters necessary for FedAvg
        return self.get_parameters(), len(self.ds_train), {loss}
    
    def evaluate(self, parameters, config):
        """Perform the eval step after having assigned new weights."""
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set model parameters from server
        self.set_parameters(parameters)
        # evaluate the model
        eval_loss = evaluating_loop(
            valloader=self.valloader,
            criterion=self.eval_criterion,
            device=device,
            autoencoder=self.autoencoder,
        )
        # returning the parameters necessary for evaluation
        return float(eval_loss), len(self.ds_test), {}
