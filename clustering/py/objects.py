#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:26:15 2021

@author: relogu
"""
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple
from tensorflow.keras.optimizers import SGD
import flwr as fl
from flwr.client import NumPyClient
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, Parameters, FitRes, Weights, parameters_to_weights
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.append('../')
import clustering.py.common_fn as my_fn
from sklearn.ensemble._hist_gradient_boosting import loss

class ClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self, autoencoder, encoder, kmeans, clustering_model, x, y, client_id,
                    pretrain_epochs: int = 100, n_clusters: int = 2, local_epochs: int = 1,
                    autoenc_local_epochs: int = 5, update_interval: int = 140):
        # set
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.pretrain_epochs = pretrain_epochs
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.clustering_model = clustering_model
        self.local_epochs = local_epochs
        self.autenc_local_epochs = autoenc_local_epochs
        self.update_interval = update_interval
        self.x = x
        self.y = y
        self.client_id = client_id
        # default
        self.f_round = 0
        self.p = None
        self.local_iter = 0
        self.step = 'autoencoder'

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step == 'autoencoder':
            return self.autoencoder.get_weights()
        elif self.step == 'k-means':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _get_step(self):
        if self.f_round < self.pretrain_epochs:
            self.step = 'autoencoder'
        elif self.f_round < self.pretrain_epochs+1:
            self.step = 'k-means'
        else :
            self.step = 'clustering'

    def _fit_clustering_model(self):
        for _ in range(int(self.local_epochs)):
            if self.local_iter % self.update_interval == 0:
                q = self.clustering_model.predict(self.x, verbose=0)
                self.p = my_fn.target_distribution(q)  # update the auxiliary target distribution p
            self.clustering_model.fit(x=self.x, y=self.p, verbose=0)
            self.local_iter += 1

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step()
        print("Federated Round number %d, step: %s" % (self.f_round, self.step))
        if self.step == 'autoencoder': # autoencoder step
            # pretrain the autoencoder
            if self.f_round == 1: # compiling
                pretrain_optimizer = SGD(lr=1, momentum=0.9)
                self.autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
            else : # setting new weights
                self.autoencoder.set_weights(parameters)
            self.autoencoder.fit(self.x, self.x, epochs=self.autenc_local_epochs, verbose=0) #, callbacks=cb)
            #self.autoencoder.save_weights('./results/ae_weights.h5')
            # returning the parameters necessary for FedAvg
            return self.autoencoder.get_weights(), len(self.x), {}
        elif self.step == 'k-means': # k-Means step
            # setting new weights
            self.autoencoder.set_weights(parameters)
            # initializing clustering model
            self.clustering_model = my_fn.create_clustering_model(self.n_clusters, self.encoder)
            # compiling the clustering model
            self.clustering_model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
            # fitting clusters' centers using k-means
            y_pred_kmeans = self.kmeans.fit_predict(self.encoder.predict(self.x))
            print('Client %d, updated accuracy of k-Means: %.5f' % (self.client_id, my_fn.acc(y, y_pred_kmeans)))
            # returning the parameters necessary for FedAvg
            return self.kmeans.cluster_centers_, len(self.x), {}
        elif self.step == 'clustering': # initialization of the the clustering model
            if self.f_round == self.pretrain_epochs+1: # initialize clustering layer with new kmeans' cluster centers
                self.clustering_model.get_layer(name='clustering').set_weights(np.array([parameters]))
            else : # getting new weights
                self.clustering_model.set_weights(parameters)
            # fitting clustering model
            self._fit_clustering_model()
            # returning the parameters necessary for FedAvg
            return self.clustering_model.get_weights(), len(self.x), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        if self.step == 'autoencoder':
            loss = self.autoencoder.evaluate(x, x, verbose=0)
            print('Client %d, FedIter %d, loss %.5f' % (self.client_id, self.f_round, loss))
        elif self.step == 'clustering':
            # Eval.
            q = self.clustering_model.predict(x, verbose=0)
            p = my_fn.target_distribution(q)  # update the auxiliary target distribution p
            loss = self.clustering_model.evaluate(x, p, verbose=0)

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if self.local_iter % 10 == 0:
                my_fn.print_confusion_matrix(y, y_pred, client_id=self.client_id)
            if y is not None:
                acc = np.round(my_fn.acc(y, y_pred), 5)
                nmi = np.round(my_fn.nmi(y, y_pred), 5)
                ari = np.round(my_fn.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Client %d, Acc = %.5f, nmi = %.5f, ari = %.5f ; loss = %.5f' % (self.client_id, acc, nmi, ari, loss))
        return loss, len(x), {"accuracy": acc}

class CommunityClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self, autoencoder, encoder, kmeans, clustering_model, x, y, client_id,
                    ae_fed_epochs: int = 1, n_clusters: int = 2, local_epochs: int = 1,
                    ae_local_epochs: int = 300, n_communities: int = 5):
        # set
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.ae_optimizer = None
        self.ae_loss = None
        self._set_ae_compiler()
        self.ae_fed_epochs = ae_fed_epochs
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.n_communities = None
        self.clustering_models = None
        self._set_clustering_models(clustering_model, n_communities)
        self.cl_optimizer = None
        self.cl_loss = None
        self._set_cl_compiler()
        self.local_epochs = local_epochs
        self.ae_local_epochs = ae_local_epochs
        self.x = x
        self.y = y
        self.client_id = client_id
        # default
        self.f_round = 0
        self.local_iter = 0
        self.step = 'autoencoder'
        self.mean_centroid = []
        self.community_weights = []
        self.fed_weights = []
    
    def _set_ae_compiler(self,
                         optimizer = SGD(lr=1, momentum=0.9),
                         loss = 'mse'):
        self.ae_optimizer = optimizer
        self.ae_loss = loss
    
    def _set_cl_compiler(self,
                         optimizer = SGD(lr=0.01, momentum=0.9), loss = 'kld'):
        self.cl_optimizer = optimizer
        self.cl_loss = loss
    
    def _set_clustering_models(self,
                                clustering_model,
                                n_communities):
        self.n_communities = n_communities
        self.clustering_models = [clustering_model]*self.n_communities
    
    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step == 'autoencoder':
            return self.autoencoder.get_weights()
        elif self.step == 'k-means':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _get_step(self, config: Dict[str, Scalar]):
        val = config['config']
        self.step = val

    def _fit_autoencoder(self, parameters):
        # pretrain the autoencoder
        if self.f_round == 1: # compiling
            self.autoencoder.compile(optimizer=self.ae_optimizer, loss=self.ae_loss)
        else : # setting new weights
            self.autoencoder.set_weights(parameters)
        self.autoencoder.fit(self.x, self.x, epochs=self.autenc_local_epochs, verbose=0) #, callbacks=cb)
        #self.autoencoder.save_weights('./results/ae_weights.h5')
        # returning the parameters necessary for FedAvg
        return self.autoencoder.get_weights(), len(self.x), {'step': self.step}
    
    def _fit_kmeans(self, parameters):
        # setting new weights
        self.autoencoder.set_weights(parameters)
        # fitting clusters' centroid using k-means
        print(self.encoder.predict(self.x)[0])
        y_pred_kmeans = self.kmeans.fit_predict(self.encoder.predict(self.x))
        print('Client %d, updated accuracy of k-Means: %.5f' % (self.client_id, my_fn.acc(self.y, y_pred_kmeans)))
        # getting the mean centroid
        print(self.kmeans.cluster_centers_)
        self.mean_centroid = np.average(self.kmeans.cluster_centers_, axis=1)
        return self.mean_centroid, len(self.x), {'step': self.step}
    
    def _fit_clustering_model(self, parameters, config):
        if config['community'] == 0 and config['round'] == 1:
            # initilizing communities k-means
            self.kmeans = KMeans(n_clusters=self.n_communities, random_state=51550)
            self.kmeans.cluster_centers_ = parameters
            y_pred_kmeans = self.kmeans.fit_predict(self.encoder.predict(self.x))
            _, self.community_weights = np.unique(y_pred_kmeans, return_counts=True)
            # getting communities' centroids
            print(self.community_weights)
            # initialize all the community model with the same weights
            for model in self.clustering_models:
                model.compile(optimizer=self.cl_optimizer, loss=self.cl_loss)
                model.load_weights('./py/my_model.h5')
        elif config['community'] > 0 and config['round'] == 1:
            # getting the final weights for the community model
            self.clustering_models[config['community']-1].set_weights(parameters)
        else:
            # getting the new weights for the community model
            self.clustering_models[config['community']].set_weights(parameters)
        # fitting the right community model
        for _ in range(self.local_epochs):
            self.clustering_models[config['community']].fit(x=self.x, y=self.y, verbose=0)
            self.local_iter += 1
        parameters_to_return = self.clustering_models[config['community']].get_weights()
        weights = self.community_weights[config['community']]
        return parameters_to_return, weights, {}

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step(config)
        print("Federated Round number %d, step: %s" % (self.f_round, self.step))
        if self.step == 'autoencoder': # autoencoder step
            # returning the parameters necessary for FedAvg
            return self._fit_autoencoder(parameters)
        elif self.step == 'k-means': # k-Means step
            return self._fit_kmeans(parameters)
        elif self.step == 'clustering': # initialization of the the clustering model
            return self._fit_clustering_model(parameters, config)

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        if self.step == 'autoencoder':
            loss = self.autoencoder.evaluate(x, x, verbose=0)
            print('Client %d, FedIter %d, loss %.5f' % (self.client_id, self.f_round, loss))
        elif self.step == 'clustering':
            # Eval.
            q = self.clustering_model.predict(x, verbose=0)
            p = my_fn.target_distribution(q)  # update the auxiliary target distribution p
            loss = self.clustering_model.evaluate(x, p, verbose=0)

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if self.local_iter % 10 == 0:
                my_fn.print_confusion_matrix(y, y_pred, client_id=self.client_id)
            if y is not None:
                acc = np.round(my_fn.acc(y, y_pred), 5)
                nmi = np.round(my_fn.nmi(y, y_pred), 5)
                ari = np.round(my_fn.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Client %d, Acc = %.5f, nmi = %.5f, ari = %.5f ; loss = %.5f' % (self.client_id, acc, nmi, ari, loss))
        return loss, len(self.x), {"accuracy": acc}