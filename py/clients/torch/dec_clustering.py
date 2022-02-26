#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
import os

import numpy as np
import torch
from torch.nn import Module, KLDivLoss
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from flwr.client import NumPyClient
from flwr.common.typing import Scalar
from typing import Union, Dict, Any, OrderedDict
from pathlib import Path

from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.dec import DEC
from py.dec.torch.utils import target_distribution, cluster_accuracy


def dec_model_training_loop(
    n_epochs: int = 1, # number of epochs
    dataloader: DataLoader = None, # dataloader for train
    device: str = 'cpu', # device to pass torch
    optimizer: Any = None, # optimizer for train
    model: Module = None, # network
):
    model.to(device)
    loss_function = KLDivLoss(reduction='sum')
    model.train()
    for _ in range(n_epochs):
        ret_loss = 0.0
        for i, batch in enumerate(dataloader):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 2:
                batch, _ = batch
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            soft_labels = output
            target = target_distribution(soft_labels).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            ret_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
        ret_loss = ret_loss / (i+1)
    return ret_loss
            
            
def dec_model_evaluating_loop(
    validation_loader: DataLoader = None, # dataloader for train
    device: str = 'cpu', # device to pass torch
    autoencoder: Module = None, # network
    model: Module = None, # dec model
):
    model.to(device)
    recon_loss = 0.0
    criterion = MSELoss()
    data = []
    r_data = []
    features = []
    prob_labels = []
    r_prob_labels = []
    actual = []
    model.eval()
    for i, batch in enumerate(validation_loader):
        with torch.no_grad():
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # unpack if we have a prediction label
                actual.append(value.cpu())
                data.append(batch.cpu())
            batch = batch.to(device, non_blocking=True)
            r_batch = autoencoder(batch)
            f_batch = autoencoder.encoder(batch)
            r_data.append(r_batch.cpu())
            features.append(f_batch.cpu())
            loss = criterion(r_batch, batch)
            recon_loss += loss.item()
            prob_labels.append(model(batch).cpu())
            r_prob_labels.append(model(r_batch).cpu())
        recon_loss = recon_loss / (i+1)
    return (
        recon_loss,
        torch.cat(prob_labels).max(1)[1].cpu().numpy(),
        torch.cat(r_prob_labels).max(1)[1].cpu().numpy(),
        torch.cat(actual).numpy(),
        torch.cat(data).numpy(),
        torch.cat(r_data).numpy(),
        torch.cat(features).numpy(),
    )


class DECClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 client_id: str,  # id of client
                 data_loader_config: Dict = None, # config of dataloader
                 net_config: Dict = None, # config of network
                 dec_config: Dict = None, # config of DEC
                 opt_config: Dict = None, # config of optimizer
                 output_folder: Union[Path, str] = None # output folder
                 ):
        self.client_id = client_id
        # set output folder        
        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)
        # get datasets
        self.ds_train = data_loader_config['get_train_fn'](client_id=eval(client_id))
        self.ds_test = data_loader_config['get_test_fn'](client_id=eval(client_id))
        self.trainloader = data_loader_config['trainloader_fn'](self.ds_train)
        self.valloader = data_loader_config['valloader_fn'](self.ds_test)
        # get network
        if 'noising' in net_config.keys():
            net_config.pop('noising')
        if 'corruption' in net_config.keys():
            net_config.pop('corruption')
        self.autoencoder = StackedDenoisingAutoEncoder(**net_config)
        # get DEC model
        self.dec_model = DEC(
            cluster_number=dec_config['n_clusters'],
            hidden_dimension=dec_config['hidden_dimension'],
            encoder=self.autoencoder.encoder,
            alpha=dec_config['alpha'])
        # get optimizer
        self.optimizer = opt_config['optimizer_fn'](
            opt_config['name'],
            opt_config['dataset'],
            opt_config['linears'],
            opt_config['lr'])(self.dec_model.parameters())
        # get previusly predicted labels
        npy_file = np.load(self.out_dir/'predicted_previous{}.npz'.format(self.client_id), allow_pickle=True)
        self.predicted_previous = np.array([npy_file[a] for a in npy_file])
        # general initializations
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
    
    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties
    
    def get_parameters(self):
        """Get the model weights by model object."""
        return [val.cpu().numpy() for _, val in self.dec_model.state_dict().items()]

    def set_ae_parameters(self, parameters):
        """Set the model weights by parameters object."""
        params_dict = zip(self.autoencoder.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.autoencoder.load_state_dict(state_dict, strict=True)
    
    def set_parameters(self, parameters):
        """Set the model weights by parameters object."""
        params_dict = zip(self.dec_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.dec_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set model parameters from server
        self.set_parameters(parameters)
        # fit DEC model
        loss = dec_model_training_loop(
            n_epochs=config['n_epochs'],
            dataloader=self.trainloader,
            device=device,
            model=self.dec_model,
            optimizer=self.optimizer,
        )
        # returning the parameters necessary for FedAvg
        return self.get_parameters(), len(self.ds_train), loss
    
    def evaluate(self, parameters, config):
        # get device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # get DEC model parameter from server
        self.set_parameters(parameters)
        # valuate clustering
        recon_loss, predicted, r_predicted, actual, data, r_data, features = dec_model_evaluating_loop(
            autoencoder=self.autoencoder,
            device=device,
            model=self.dec_model,
            validation_loader=self.valloader,
        )
        # TODO: evaluate reconstruction
        # evaluate clustering
        delta_label = (
            np.sum(predicted != self.predicted_previous)
            / self.predicted_previous.shape[0]
        )
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
        r_reassignment, cycle_accuracy = cluster_accuracy(r_predicted, predicted)
        # TODO: save metrics
        # save labels for predicted_previous of next step
        np.savez(self.out_dir/'predicted_previous{}.npz'.format(self.client_id), *predicted)
        # returning the parameters necessary for evaluation
        return float(accuracy), len(self.ds_test), {'cycle accuracy': float(cycle_accuracy),
                                                    'delta label': float(delta_label),
                                                    'cosine silhouette score': float(cos_sil_score),
                                                    'euclidean silhouette score': float(eucl_sil_score),
                                                    'data calinski harabasz score': float(data_calinski_harabasz),
                                                    'features calinski harabasz score': float(feat_calinski_harabasz)}
