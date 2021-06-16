#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""

from sklearn.ensemble._hist_gradient_boosting import loss
import clustering.py.common_fn as my_fn
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple
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
sys.path.append('../')

k_means_initializer = 'k-means++'
k_means_eval_string = 'Client %d, updated real accuracy of k-Means: %.5f'
out_1 = 'Client %d, FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f'
clustering_eval_string = 'Client %d, Acc = %.5f, nmi = %.5f, ari = %.5f ; loss = %.5f'


class SimpleClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self, autoencoder, encoder, x, y, client_id,
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
        self.x = x
        self.y = y
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
        elif self.step == 'k-FED':
            return self.kmeans.cluster_centers_
        elif self.step == 'pretrain_ae':
            return self.autoencoder.get_weights()
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

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
            self.autoencoder.fit(x=self.x,
                                 y=self.x,
                                 batch_size=32,
                                 epochs=self.ae_local_epochs)
            # returning the parameters necessary for FedAvg
            return self.autoencoder.get_weights(), len(self.x), {}
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
            self.kmeans.fit(self.encoder.predict(self.x))
            # returning the parameters necessary for k-FED
            return self.kmeans.cluster_centers_, len(self.x), {}
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
            return self.clustering_model.get_weights(), len(self.x), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = ()
        metrics = {}
        if self.step == 'k-FED':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(self.encoder.predict(self.x))
            # computing metrics
            acc = my_fn.acc(self.y, y_pred_kmeans)
            nmi = my_fn.nmi(self.y, y_pred_kmeans)
            ami = my_fn.ami(self.y, y_pred_kmeans)
            ari = my_fn.ari(self.y, y_pred_kmeans)
            ran = my_fn.ran(self.y, y_pred_kmeans)
            homo = my_fn.homo(self.y, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                my_fn.print_confusion_matrix(
                    self.y, y_pred_kmeans, client_id=self.client_id)
            # retrieving the results
            result = (loss, len(self.x), metrics)
        elif self.step == 'pretrain_ae':
            # evaluation
            loss = self.autoencoder.evaluate(self.x, self.x, verbose=0)
            result = (loss, len(self.x), {})
        elif self.step == 'clustering':
            # evaluation
            q = self.clustering_model.predict(self.x, verbose=0)
            # update the auxiliary target distribution p
            p = my_fn.target_distribution(q)
            # retrieving loss
            loss = self.clustering_model.evaluate(self.x, p, verbose=0)
            # evaluate the clustering performance using some metrics
            y_pred = q.argmax(1)
            if self.y is not None:
                acc = my_fn.acc(self.y, y_pred)
                nmi = my_fn.nmi(self.y, y_pred)
                ami = my_fn.ami(self.y, y_pred)
                ari = my_fn.ari(self.y, y_pred)
                ran = my_fn.ran(self.y, y_pred)
                homo = my_fn.homo(self.y, y_pred)
                if self.f_round % 10 == 0:  # print confusion matrix
                    my_fn.print_confusion_matrix(
                        self.y, y_pred, client_id=self.client_id)
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
            result = (loss, len(self.x), metrics)
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
        self.x = x
        self.y = y
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
            y_pred_kmeans = self.kmeans.fit_predict(self.x)
            print(k_means_eval_string %
                  (self.client_id, my_fn.acc(self.y, y_pred_kmeans)))
            # returning the parameters necessary for FedAvg
            return self.kmeans.cluster_centers_, len(self.x), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        if self.step == 'k-means':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(self.x)
            # computing metrics
            acc = my_fn.acc(self.y, y_pred_kmeans)
            nmi = my_fn.nmi(self.y, y_pred_kmeans)
            ami = my_fn.ami(self.y, y_pred_kmeans)
            ari = my_fn.ari(self.y, y_pred_kmeans)
            ran = my_fn.ran(self.y, y_pred_kmeans)
            homo = my_fn.homo(self.y, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                my_fn.print_confusion_matrix(
                    self.y, y_pred_kmeans, client_id=self.client_id)
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
            result = (loss, len(self.x), metrics)
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
                         optimizer=SGD(lr=1, momentum=0.9),
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
            self.x, self.x, epochs=self.autenc_local_epochs, verbose=0)
        # self.autoencoder.save_weights('./results/ae_weights.h5')
        # returning the parameters necessary for FedAvg
        return self.autoencoder.get_weights(), len(self.x), {'step': self.step}

    def _fit_kmeans(self, parameters):
        # setting new weights
        self.autoencoder.set_weights(parameters)
        # fitting clusters' centroid using k-means
        print(self.encoder.predict(self.x)[0])
        y_pred_kmeans = self.kmeans.fit_predict(self.encoder.predict(self.x))
        print('Client %d, updated accuracy of k-Means: %.5f' %
              (self.client_id, my_fn.acc(self.y, y_pred_kmeans)))
        # getting the mean centroid
        print(self.kmeans.cluster_centers_)
        self.mean_centroid = np.average(self.kmeans.cluster_centers_, axis=1)
        return self.mean_centroid, len(self.x), {'step': self.step}

    def _fit_clustering_model(self, parameters, config):
        if config['community'] == 0 and config['round'] == 1:
            # initilizing communities k-means
            self.kmeans = KMeans(
                n_clusters=self.n_communities, random_state=51550)
            self.kmeans.cluster_centers_ = parameters
            y_pred_kmeans = self.kmeans.fit_predict(
                self.encoder.predict(self.x))
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
                x=self.x, y=self.y, verbose=0)
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
            loss = self.autoencoder.evaluate(x, x, verbose=0)
        elif self.step == 'clustering':
            # Eval.
            q = self.clustering_model.predict(x, verbose=0)
            # update the auxiliary target distribution p
            p = my_fn.target_distribution(q)
            loss = self.clustering_model.evaluate(x, p, verbose=0)

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if self.local_iter % 10 == 0:
                my_fn.print_confusion_matrix(
                    y, y_pred, client_id=self.client_id)
            if y is not None:
                acc = np.round(my_fn.acc(y, y_pred), 5)
                nmi = np.round(my_fn.nmi(y, y_pred), 5)
                ari = np.round(my_fn.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print(clustering_eval_string %
                      (self.client_id, acc, nmi, ari, loss))
        return loss, len(self.x), {"accuracy": acc}
