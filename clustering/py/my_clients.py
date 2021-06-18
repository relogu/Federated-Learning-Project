#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
import clustering.py.common_fn as my_fn
from clustering.py.clustergan import Generator_CNN, Discriminator_CNN, Encoder_CNN, sample_z, calc_gradient_penalty
import os
from sklearn.ensemble._hist_gradient_boosting import loss
from functools import reduce
from itertools import chain as ichain
from typing import Callable, Dict, List, Optional, Tuple, OrderedDict
from tensorflow.keras.optimizers import SGD
import flwr as fl
from flwr.client import NumPyClient
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, Parameters, FitRes, Weights, parameters_to_weights
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans
import math
import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
sys.path.append('../')

k_means_initializer = 'k-means++'
k_means_eval_string = 'Client %d, updated real accuracy of k-Means: %.5f'
out_1 = 'Client %d, FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f'
clustering_eval_string = 'Client %d, Acc = %.5f, nmi = %.5f, ari = %.5f ; loss = %.5f'


class SimpleClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 autoencoder,
                 encoder,
                 x,
                 y,
                 client_id,
                 kmeans_local_epochs: int = 1,
                 ae_local_epochs: int = 1,
                 cl_local_epochs: int = 1,
                 update_interval: int = 55,
                 n_clusters: int = 2,
                 seed: int = 51550):
        # set
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.clustering_model = None
        self.kmeans_local_epochs = kmeans_local_epochs
        self.cl_local_epochs = cl_local_epochs
        self.ae_local_epochs = ae_local_epochs
        self.update_interval = update_interval
        self.x = x
        self.y = y
        self.client_id = client_id
        self.seed = seed
        # default
        self.f_round = 0
        self.p = None
        self.local_iter = 0
        self.step = None

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:
            return []
        elif self.step == 'autoencoder':
            return self.autoencoder.get_weights()
        elif self.step == 'k-means':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _get_step(self, config):
        self.step = config['model']

    def _fit_clustering_model(self):
        for _ in range(int(self.cl_local_epochs)):
            if self.local_iter % self.update_interval == 0:
                q = self.clustering_model.predict(self.x, verbose=0)
                # update the auxiliary target distribution p
                self.p = my_fn.target_distribution(q)
            self.clustering_model.fit(x=self.x, y=self.p, verbose=0)
            self.local_iter += 1

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step(config)
        print("Federated Round number %d, step: %s, rounds %d/%d, is first %s" % (self.f_round,
                                                                                  self.step,
                                                                                  config['actual_round'],
                                                                                  config['total_round'],
                                                                                  str(config['first'])))
        if self.step == 'autoencoder':  # autoencoder step
            # pretrain the autoencoder
            if config['first']:  # compiling
                pretrain_optimizer = SGD(lr=1, momentum=0.9)
                self.autoencoder.compile(
                    optimizer=pretrain_optimizer, loss='mse')
            else:  # setting new weights
                self.autoencoder.set_weights(parameters)
            # , callbacks=cb)
            self.autoencoder.fit(
                self.x, self.x, epochs=self.ae_local_epochs, verbose=0)
            # self.autoencoder.save_weights('./results/ae_weights.h5')
            # returning the parameters necessary for FedAvg
            return self.autoencoder.get_weights(), len(self.x), {}
        elif self.step == 'k-means':  # k-Means step
            if config['first']:
                parameters = k_means_initializer
                n_init = 25
            else:
                parameters = np.array(parameters)
                n_init = 1
            self.kmeans = KMeans(init=parameters, n_clusters=self.n_clusters,
                                 max_iter=self.kmeans_local_epochs, n_init=n_init,
                                 random_state=self.seed)
            # getting predictions
            predictions = self.encoder.predict(self.x)
            # fitting clusters' centers using k-means
            self.kmeans.fit(predictions)
            # returning the parameters necessary for FedAvg
            return self.kmeans.cluster_centers_, len(self.x), {}
        elif self.step == 'clustering':  # initialization of the the clustering model
            if config['first']:  # initialize clustering layer with new kmeans' cluster centers
                # initializing clustering model
                self.clustering_model = my_fn.create_clustering_model(
                    self.n_clusters, self.encoder)
                # compiling the clustering model
                self.clustering_model.compile(
                    optimizer=SGD(0.01, 0.9), loss='kld')
                self.clustering_model.get_layer(
                    name='clustering').set_weights(np.array([parameters]))
            else:  # getting new weights
                self.clustering_model.set_weights(parameters)
            # fitting clustering model
            self._fit_clustering_model()
            # returning the parameters necessary for FedAvg
            return self.clustering_model.get_weights(), len(self.x), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = ()
        if self.step == 'autoencoder':
            loss = self.autoencoder.evaluate(self.x, self.x, verbose=0)
            print('Client %d, FedIter %d, loss %.5f' %
                  (self.client_id, self.f_round, loss))
            result = (loss, len(self.x), {})
        elif self.step == 'k-means':
            predictions = self.encoder.predict(self.x)
            y_pred_kmeans = self.kmeans.fit_predict(predictions)
            acc = my_fn.acc(self.y, y_pred_kmeans)
            print(k_means_eval_string % (self.client_id, acc))
            result = (loss, len(self.x), {"accuracy": acc})
        elif self.step == 'clustering':
            # Eval.
            q = self.clustering_model.predict(self.x, verbose=0)
            # update the auxiliary target distribution p
            p = my_fn.target_distribution(q)
            loss = self.clustering_model.evaluate(self.x, p, verbose=0)
            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if (self.f_round-1) % 10 == 0:
                my_fn.print_confusion_matrix(
                    self.y, y_pred, client_id=self.client_id)
            if self.y is not None:
                acc = np.round(my_fn.acc(self.y, y_pred), 5)
                nmi = np.round(my_fn.nmi(self.y, y_pred), 5)
                ari = np.round(my_fn.ari(self.y, y_pred), 5)
                loss = np.round(loss, 5)
                print(clustering_eval_string %
                      (self.client_id, acc, nmi, ari, loss))
                result = (loss, len(self.x), {
                          "accuracy": acc, "norm_mutual_info": nmi, "adj_random_index": ari})
        return result


class KFEDClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 x,  # data
                 y,  # labels
                 client_id,  # id of the client
                 ae_dims,  # dimensions of encoder's dense layers
                 ae_local_epochs: int = 1,  # number of local epochs for autoencoder pretrain
                 cl_local_epochs: int = 1,  # number of local epochs for clustering step
                 update_interval: int = 55,  # update interval to refresh t-distribution
                 n_clusters: int = 2,  # number of total clusters
                 seed: int = 51550):  # see for random gen
        # set
        self.n_clusters = n_clusters
        self.ae_dims = ae_dims
        self.ae_local_epochs = ae_local_epochs
        self.cl_local_epochs = cl_local_epochs
        self.update_interval = update_interval
        self.x_train, self.y_train, self.x_test, self.y_test = my_fn.split_dataset(
            x, y)
        self.client_id = client_id
        self.seed = seed
        # default
        self.autoencoder = None
        self.encoder = None
        self.clustering_model = None
        self.f_round = 0
        self.p = None
        self.local_iter = 0
        self.step = None
        self.cluster_centers = None

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:
            return []
        elif self.step == 'pretrain_ae':
            return self.autoencoder.get_weights()
        elif self.step == 'k-FED':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _fit_clustering_model(self):
        for _ in range(int(self.cl_local_epochs)):
            if self.local_iter % self.update_interval == 0:
                q = self.clustering_model.predict(self.x_train, verbose=0)
                # update the auxiliary target distribution p
                self.p = my_fn.target_distribution(q)
            self.clustering_model.fit(x=self.x_train, y=self.p, verbose=0)
            self.local_iter += 1

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self.step = config['model']
        print("Federated Round number %d, step: %s, rounds %d/%d" % (self.f_round,
                                                                     self.step,
                                                                     config['actual_round'],
                                                                     config['total_rounds']))
        if self.step == 'pretrain_ae':  # ae pretrain step
            if config['first']:
                # building and compiling autoencoder
                self.autoencoder, self.encoder = my_fn.create_autoencoder(
                    self.ae_dims)
                self.autoencoder.compile(
                    optimizer=SGD(lr=0.001, momentum=0.9),
                    loss='mse'
                )
            else:  # getting new weights
                self.autoencoder.set_weights(parameters)
            # fitting the autoencoder
            self.autoencoder.fit(x=self.x_train,
                                 y=self.x_train,
                                 batch_size=32,
                                 epochs=self.ae_local_epochs)
            # returning the parameters necessary for FedAvg
            return self.autoencoder.get_weights(), len(self.x_train), {}
        elif self.step == 'k-FED':  # k-Means step
            parameters = k_means_initializer  # k++ is used
            # number of different random initializations
            n_init = 25
            # maximum total local epochs for k-means algorithm (may be relaxed/removed)
            local_epochs = 300
            # number of cluster following the definition of heterogeneity
            n_cl = int(math.sqrt(self.n_clusters))
            self.kmeans = KMeans(init=parameters,
                                 n_clusters=n_cl,
                                 max_iter=local_epochs,
                                 n_init=n_init,
                                 random_state=self.seed)
            # fitting clusters' centers using k-means
            self.kmeans.fit(self.encoder.predict(self.x_train))
            # returning the parameters necessary for k-FED
            return self.kmeans.cluster_centers_, len(self.x_train), {}
        elif self.step == 'clustering':
            if config['first']:  # initialize clustering layer with final kmeans' cluster centers
                # getting final clusters centers
                self.cluster_centers = parameters
                # initializing clustering model
                self.clustering_model = my_fn.create_clustering_model(
                    self.n_clusters,
                    self.encoder)
                # compiling the clustering model
                self.clustering_model.compile(
                    optimizer=SGD(0.01, 0.9),
                    loss='kld')
                self.clustering_model.get_layer(
                    name='clustering').set_weights(np.array([self.cluster_centers]))
            else:  # getting new weights
                self.clustering_model.set_weights(parameters)
            # fitting clustering model
            self._fit_clustering_model()
            # returning the parameters necessary for FedAvg
            return self.clustering_model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = ()
        metrics = {}
        if self.step == 'pretrain_ae':
            # evaluation
            loss = self.autoencoder.evaluate(
                self.x_test, self.x_test, verbose=0)
            metrics = {"loss": loss}
            result = metrics.copy()
            result['client'] = self.client_id
            result['round'] = self.f_round
            my_fn.dump_result_dict('client_'+str(self.client_id)+'_ae', result)
            result = (loss, len(self.x_test), {})
        elif self.step == 'k-FED':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(
                self.encoder.predict(self.x_test))
            # computing metrics
            acc = my_fn.acc(self.y_test, y_pred_kmeans)
            nmi = my_fn.nmi(self.y_test, y_pred_kmeans)
            ami = my_fn.ami(self.y_test, y_pred_kmeans)
            ari = my_fn.ari(self.y_test, y_pred_kmeans)
            ran = my_fn.ran(self.y_test, y_pred_kmeans)
            homo = my_fn.homo(self.y_test, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                my_fn.print_confusion_matrix(
                    self.y_test, y_pred_kmeans, client_id=self.client_id)
            # retrieving the results
            result = (loss, len(self.x_test), metrics)
        elif self.step == 'clustering':
            # evaluation
            q = self.clustering_model.predict(self.x_test, verbose=0)
            # update the auxiliary target distribution p
            p = my_fn.target_distribution(q)
            # retrieving loss
            loss = self.clustering_model.evaluate(self.x_test, p, verbose=0)
            # evaluate the clustering performance using some metrics
            y_pred = q.argmax(1)
            if self.y_test is not None:
                acc = my_fn.acc(self.y_test, y_pred)
                nmi = my_fn.nmi(self.y_test, y_pred)
                ami = my_fn.ami(self.y_test, y_pred)
                ari = my_fn.ari(self.y_test, y_pred)
                ran = my_fn.ran(self.y_test, y_pred)
                homo = my_fn.homo(self.y_test, y_pred)
                if self.f_round % 10 == 0:  # print confusion matrix
                    my_fn.print_confusion_matrix(
                        self.y_test, y_pred, client_id=self.client_id)
                print(out_1 % (self.client_id, self.f_round,
                               acc, nmi, ami, ari, ran, homo))
                # dumping and retrieving the results
                metrics = {"accuracy": acc,
                           "normalized_mutual_info_score": nmi,
                           "adjusted_mutual_info_score": ami,
                           "adjusted_rand_score": ari,
                           "rand_score": ran,
                           "homogeneity_score": homo}
                result = metrics.copy()
                result['loss'] = loss
                result['client'] = self.client_id
                result['round'] = self.local_iter
                my_fn.dump_result_dict('client_'+str(self.client_id), result)
            result = (loss, len(self.x_test), metrics)
        return result


class SimpleKMeansClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 x,
                 y,
                 client_id,
                 n_clusters: int = 2,
                 seed: int = 51550,
                 kmeans_local_epochs: int = 1,
                 model_local_epochs: int = 1):
        # set
        self.x_train, self.y_train, self.x_test, self.y_test = my_fn.split_dataset(
            x, y)
        self.client_id = client_id
        self.seed = seed
        self.kmeans_local_epochs = kmeans_local_epochs
        self.n_clusters = n_clusters
        # default
        self.kmeans = None
        self.f_round = 0
        self.p = None
        self.step = None

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:  # first federated iteration
            return []
        elif self.step == 'k-means':
            return self.kmeans.cluster_centers_

    def _get_step(self, config):
        self.step = config['model']

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step(config)
        print("Federated Round number %d, step: %s" %
              (self.f_round, self.step))
        if self.step == 'k-means':  # k-Means step
            if parameters == []:
                parameters = k_means_initializer
                n_init = 25
            else:
                parameters = np.array(parameters)
                n_init = 1
            self.kmeans = KMeans(init=parameters,
                                 n_clusters=self.n_clusters,
                                 max_iter=self.kmeans_local_epochs,
                                 n_init=n_init,
                                 random_state=self.seed)
            # fitting clusters' centers using k-means
            y_pred_kmeans = self.kmeans.fit_predict(self.x_train)
            print(k_means_eval_string %
                  (self.client_id, my_fn.acc(self.y_train, y_pred_kmeans)))
            # returning the parameters necessary for FedAvg
            return self.kmeans.cluster_centers_, len(self.x_train), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        if self.step == 'k-means':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(self.x_test)
            # computing metrics
            acc = my_fn.acc(self.y_test, y_pred_kmeans)
            nmi = my_fn.nmi(self.y_test, y_pred_kmeans)
            ami = my_fn.ami(self.y_test, y_pred_kmeans)
            ari = my_fn.ari(self.y_test, y_pred_kmeans)
            ran = my_fn.ran(self.y_test, y_pred_kmeans)
            homo = my_fn.homo(self.y_test, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                my_fn.print_confusion_matrix(
                    self.y_test, y_pred_kmeans, client_id=self.client_id)
            # dumping and retrieving the results
            metrics = {"accuracy": acc,
                       "normalized_mutual_info_score": nmi,
                       "adjusted_mutual_info_score": ami,
                       "adjusted_rand_score": ari,
                       "rand_score": ran,
                       "homogeneity_score": homo}
            result = metrics.copy()
            result['client'] = self.client_id
            result['round'] = self.f_round
            my_fn.dump_result_dict('client_'+str(self.client_id), result)
            result = (loss, len(self.x_test), metrics)
        return result


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
        self.x_train, self.y_train, self.x_test, self.y_test = my_fn.split_dataset(
            x, y)
        self.client_id = client_id
        # default
        self.f_round = 0
        self.local_iter = 0
        self.step = 'autoencoder'
        self.mean_centroid = []
        self.community_weights = []
        self.fed_weights = []

    def _set_ae_compiler(self,
                         optimizer=SGD(lr=0.01, momentum=0.9),
                         loss='mse'):
        self.ae_optimizer = optimizer
        self.ae_loss = loss

    def _set_cl_compiler(self,
                         optimizer=SGD(lr=0.01, momentum=0.9), loss='kld'):
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
        if self.f_round == 1:  # compiling
            self.autoencoder.compile(
                optimizer=self.ae_optimizer, loss=self.ae_loss)
        else:  # setting new weights
            self.autoencoder.set_weights(parameters)
        # , callbacks=cb)
        self.autoencoder.fit(
            self.x_train, self.x_train, epochs=self.autenc_local_epochs, verbose=0)
        # self.autoencoder.save_weights('./results/ae_weights.h5')
        # returning the parameters necessary for FedAvg
        return self.autoencoder.get_weights(), len(self.x_train), {'step': self.step}

    def _fit_kmeans(self, parameters):
        # setting new weights
        self.autoencoder.set_weights(parameters)
        # fitting clusters' centroid using k-means
        print(self.encoder.predict(self.x_train)[0])
        y_pred_kmeans = self.kmeans.fit_predict(
            self.encoder.predict(self.x_train))
        print('Client %d, updated accuracy of k-Means: %.5f' %
              (self.client_id, my_fn.acc(self.y_train, y_pred_kmeans)))
        # getting the mean centroid
        print(self.kmeans.cluster_centers_)
        self.mean_centroid = np.average(self.kmeans.cluster_centers_, axis=1)
        return self.mean_centroid, len(self.x_train), {'step': self.step}

    def _fit_clustering_model(self, parameters, config):
        if config['community'] == 0 and config['round'] == 1:
            # initilizing communities k-means
            self.kmeans = KMeans(
                n_clusters=self.n_communities, random_state=51550)
            self.kmeans.cluster_centers_ = parameters
            y_pred_kmeans = self.kmeans.fit_predict(
                self.encoder.predict(self.x_train))
            _, self.community_weights = np.unique(
                y_pred_kmeans, return_counts=True)
            # getting communities' centroids
            print(self.community_weights)
            # initialize all the community model with the same weights
            for model in self.clustering_models:
                model.compile(optimizer=self.cl_optimizer, loss=self.cl_loss)
                model.load_weights('./py/my_model.h5')
        elif config['community'] > 0 and config['round'] == 1:
            # getting the final weights for the community model
            self.clustering_models[config['community'] -
                                   1].set_weights(parameters)
        else:
            # getting the new weights for the community model
            self.clustering_models[config['community']].set_weights(parameters)
        # fitting the right community model
        for _ in range(self.local_epochs):
            self.clustering_models[config['community']].fit(
                x=self.x_train, y=self.y_train, verbose=0)
            self.local_iter += 1
        parameters_to_return = self.clustering_models[config['community']].get_weights(
        )
        weights = self.community_weights[config['community']]
        return parameters_to_return, weights, {}

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step(config)
        print("Federated Round number %d, step: %s" %
              (self.f_round, self.step))
        if self.step == 'autoencoder':  # autoencoder step
            # returning the parameters necessary for FedAvg
            return self._fit_autoencoder(parameters)
        elif self.step == 'k-means':  # k-Means step
            return self._fit_kmeans(parameters)
        elif self.step == 'clustering':  # initialization of the the clustering model
            return self._fit_clustering_model(parameters, config)

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        if self.step == 'autoencoder':
            loss = self.autoencoder.evaluate(
                self.x_test, self.x_test, verbose=0)
        elif self.step == 'clustering':
            # Eval.
            q = self.clustering_model.predict(self.x_test, verbose=0)
            # update the auxiliary target distribution p
            p = my_fn.target_distribution(q)
            loss = self.clustering_model.evaluate(self.x_test, p, verbose=0)

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if self.local_iter % 10 == 0:
                my_fn.print_confusion_matrix(
                    self.y_test, y_pred, client_id=self.client_id)
            if self.y_test is not None:
                acc = np.round(my_fn.acc(self.y_test, y_pred), 5)
                nmi = np.round(my_fn.nmi(self.y_test, y_pred), 5)
                ari = np.round(my_fn.ari(self.y_test, y_pred), 5)
                loss = np.round(loss, 5)
                print(clustering_eval_string %
                      (self.client_id, acc, nmi, ari, loss))
        return loss, len(self.x_test), {"accuracy": acc}


class ClusterGANClient(NumPyClient):

    def __init__(self,
                 x,
                 y,
                 config,
                 client_id: int = 0
                 ):
        # Training details
        self.n_epochs = config['n_local_epochs']
        self.lr = config['learning_rate']
        self.b1 = config['beta_1']
        self.b2 = config['beta_2']
        self.decay = config['decay']
        self.n_skip_iter = config['d_step']

        # Data dimensions
        self.x_shape = x.shape[1:]
        # Latent space info
        self.latent_dim = config['latent_dim']
        self.n_c = config['n_clusters']
        self.betan = config['betan']
        self.betac = config['betac']

        # Wasserstein+GP metric flag
        self.wass_metric = config['wass_metric']
        print('Using metric {}'.format(
            'Wassestrain' if self.wass_metric else 'Vanilla'))

        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        print('Using device {}'.format(self.device))
        torch.autograd.set_detect_anomaly(True)

        # Loss function
        self.bce_loss = torch.nn.BCELoss()
        self.xe_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        # Initialize generator and discriminator
        self.generator = Generator_CNN(self.latent_dim,
                                       self.n_c,
                                       self.x_shape)
        self.encoder = Encoder_CNN(self.latent_dim,
                                   self.n_c)
        self.discriminator = Discriminator_CNN(wass_metric=self.wass_metric)

        if self.cuda:
            self.generator.cuda()
            self.encoder.cuda()
            self.discriminator.cuda()
            self.bce_loss.cuda()
            self.xe_loss.cuda()
            self.mse_loss.cuda()
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Configure data loader
        self.batch_size = config['batch_size']
        self.x_train, self.y_train, self.x_test, self.y_test = my_fn.split_dataset(
            x, y)
        self.trainloader = DataLoader(
            my_fn.PrepareData(self.x_train, y=self.y_train),
            batch_size=self.batch_size)
        self.testloader = DataLoader(
            my_fn.PrepareData(self.x_test, y=self.y_test),
            batch_size=self.batch_size)
        self.test_imgs, self.test_labels = next(iter(self.testloader))
        self.test_imgs = Variable(self.test_imgs.type(self.Tensor))

        self.ge_chain = ichain(self.generator.parameters(),
                               self.encoder.parameters())

        self.optimizer_GE = torch.optim.Adam(self.ge_chain,
                                             lr=self.lr,
                                             betas=(self.b1, self.b2),
                                             weight_decay=self.decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr,
                                            betas=(self.b1, self.b2))

        # ----------
        #  Training
        # ----------
        self.ge_l = []
        self.d_l = []
        self.c_zn = []
        self.c_zc = []
        self.c_i = []
        
        # metrics
        self.img_mse_loss = None
        self.lat_mse_loss = None
        self.lat_xe_loss = None

        # leghts of NN parameters to send and receive
        self.g_w_l = len(self.generator.state_dict().items())
        self.d_w_l = len(self.discriminator.state_dict().items())
        self.e_w_l = len(self.encoder.state_dict().items())

        # initiliazing to zero the federated epochs counter
        self.f_epoch = 0

        # for saving images
        self.client_id = client_id
        self.dir_to_save_images = 'client_%d_images' % (self.client_id)
        os.makedirs(self.dir_to_save_images, exist_ok=True)

    def train(self, config):
        # Training loop
        print('\nBegin training session with %i epochs...\n' % (self.n_epochs))
        for epoch in range(self.n_epochs):
            for i, (imgs, self.itruth_label) in enumerate(self.trainloader):

                # Ensure generator/encoder are trainable
                self.generator.train()
                self.encoder.train()

                # Zero gradients for models, resetting at each iteration because they sum up,
                # and we don't want them to pile up between different iterations
                self.generator.zero_grad()
                self.encoder.zero_grad()
                self.discriminator.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_GE.zero_grad()

                # Configure input
                self.real_imgs = Variable(imgs.type(self.Tensor))

                # ---------------------------
                #  Train Generator + Encoder
                # ---------------------------

                # Sample random latent variables
                zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                          latent_dim=self.latent_dim,
                                          n_c=self.n_c)

                # Generate a batch of images
                gen_imgs = self.generator(zn, zc)

                # Discriminator output from real and generated samples
                gen_d = self.discriminator(gen_imgs)
                real_d = self.discriminator(self.real_imgs)
                valid = Variable(self.Tensor(gen_imgs.size(
                    0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(gen_imgs.size(
                    0), 1).fill_(0.0), requires_grad=False)

                # Step for Generator & Encoder, n_skip_iter times less than for discriminator
                if (i % self.n_skip_iter == 0):
                    # Encode the generated images
                    enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = self.encoder(
                        gen_imgs)

                    # Calculate losses for z_n, z_c
                    zn_loss = self.mse_loss(enc_gen_zn, zn)
                    zc_loss = self.xe_loss(enc_gen_zc_logits, zc_idx)

                    # Check requested metric
                    if self.wass_metric:
                        # Wasserstein GAN loss
                        # ge_loss = torch.mean(gen_d) + betan * zn_loss + betac * zc_loss # original
                        # corrected
                        ge_loss = - \
                            torch.mean(gen_d) + self.betan * \
                            zn_loss + self.betac * zc_loss
                    else:
                        # Vanilla GAN loss
                        v_loss = self.bce_loss(gen_d, valid)
                        ge_loss = v_loss + self.betan * zn_loss + self.betac * zc_loss
                    # backpropagate the gradients
                    ge_loss.backward(retain_graph=True)
                    # computes the new weights
                    self.optimizer_GE.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Measure discriminator's ability to classify real from generated samples
                if self.wass_metric:
                    # Gradient penaltytorch.autograd.set_detect_anomaly(True) term
                    grad_penalty = calc_gradient_penalty(
                        self.discriminator, self.real_imgs, gen_imgs)

                    # Wasserstein GAN loss w/gradient penalty
                    # d_losss = torch.mean(real_d) - torch.mean(gen_d) + grad_penalty # original
                    # corrected
                    d_loss = - torch.mean(real_d) + \
                        torch.mean(gen_d) + grad_penalty

                else:
                    # Vanilla GAN loss
                    real_loss = self.bce_loss(real_d, valid)
                    fake_loss = self.bce_loss(gen_d, fake)
                    d_loss = (real_loss + fake_loss) / 2

                d_loss.backward(inputs=list(self.discriminator.parameters()))
                self.optimizer_D.step()
            # Save training losses
            self.d_l.append(d_loss.item())
            self.ge_l.append(ge_loss.item())
            print("[Federated Epoch %d/%d] [Client ID %d] [Epoch %d/%d] \n"
                  "\tModel Losses: [D: %f] [GE: %f]" %
                  (self.f_epoch,
                   config['total_epochs'],
                   self.client_id,
                   epoch,
                   self.n_epochs,
                   d_loss.item(),
                   ge_loss.item())
                  )

    def test(self, config):
        # Generator in eval mode
        self.generator.eval()
        self.encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        # Cycle through test real -> enc -> gen
        t_imgs, t_label = self.test_imgs.data, self.test_labels
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = self.encoder(t_imgs)
        # Generate sample instances from encoding
        teg_imgs = self.generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        self.img_mse_loss = self.mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        self.c_i.append(self.img_mse_loss.item())

        # Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=self.latent_dim,
                                                 n_c=self.n_c)
        # Generate sample instances
        gen_imgs_samp = self.generator(zn_samp, zc_samp)

        # Encode sample instances
        zn_e, zc_e, zc_e_logits = self.encoder(gen_imgs_samp)

        # Calculate cycle latent losses
        self.lat_mse_loss = self.mse_loss(zn_e, zn_samp)
        self.lat_xe_loss = self.xe_loss(zc_e_logits, zc_samp_idx)

        # Save latent space cycle losses
        self.c_zn.append(self.lat_mse_loss.item())
        self.c_zc.append(self.lat_xe_loss.item())

        # Save cycled and generated examples!
        r_imgs, i_label = self.real_imgs.data[:
                                              n_samp], self.itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = self.encoder(r_imgs)
        reg_imgs = self.generator(e_zn, e_zc)
        save_image(reg_imgs.data[:n_samp],
                   self.dir_to_save_images +
                   '/cycle_reg_%06i.png' % (self.f_epoch),
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp],
                   self.dir_to_save_images+'/gen_%06i.png' % (self.f_epoch),
                   nrow=n_sqrt_samp, normalize=True)

        # Generate samples for specified classes
        stack_imgs = []
        for idx in range(self.n_c):
            # Sample specific class
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=self.n_c,
                                                     latent_dim=self.latent_dim,
                                                     n_c=self.n_c,
                                                     fix_class=idx)

            # Generate sample instances
            gen_imgs_samp = self.generator(zn_samp, zc_samp)

            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        save_image(stack_imgs,
                   self.dir_to_save_images +
                   '/gen_classes_%06i.png' % (self.f_epoch),
                   nrow=self.n_c, normalize=True)

        print("[Federated Epoch %d/%d] [Client ID %d] \n"
              "\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]" %
              (self.f_epoch,
               config['total_epochs'],
               self.client_id,
               self.img_mse_loss.item(),
               self.lat_mse_loss.item(),
               self.lat_xe_loss.item())
              )

    def get_parameters(self):
        g_par = [val.cpu().numpy()
                 for _, val in self.generator.state_dict().items()]
        d_par = [val.cpu().numpy()
                 for _, val in self.discriminator.state_dict().items()]
        e_par = [val.cpu().numpy()
                 for _, val in self.encoder.state_dict().items()]
        parameters = np.concatenate([g_par, d_par, e_par], axis=0)
        return parameters

    def set_parameters(self, parameters):
        # generator
        g_par = parameters[:self.g_w_l].copy()
        params_dict = zip(self.generator.state_dict().keys(), g_par)
        g_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # discriminator
        d_par = parameters[self.g_w_l:int(self.g_w_l+self.d_w_l)].copy()
        params_dict = zip(self.discriminator.state_dict().keys(), d_par)
        d_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # encoder
        e_par = parameters[int(self.g_w_l+self.d_w_l):].copy()
        params_dict = zip(self.encoder.state_dict().keys(), e_par)
        e_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # checking for null weights
        g_state_dict = my_fn.check_weights_dict(g_state_dict)
        d_state_dict = my_fn.check_weights_dict(d_state_dict)
        e_state_dict = my_fn.check_weights_dict(e_state_dict)
        # assigning weights
        self.generator.load_state_dict(g_state_dict, strict=True)
        self.discriminator.load_state_dict(d_state_dict, strict=True)
        self.encoder.load_state_dict(e_state_dict, strict=True)

    def fit(self, parameters, config):
        self.f_epoch += 1
        self.set_parameters(parameters)
        self.train(config)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.test(config)
        metrics = {"x": self.img_mse_loss.item(),
                   "z_n": self.lat_mse_loss.item(),
                   "z_c": self.lat_xe_loss.item()}
        return float(self.img_mse_loss.item()), len(self.testloader), metrics
