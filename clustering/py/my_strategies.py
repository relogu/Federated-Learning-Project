#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple
from tensorflow.keras.optimizers import SGD
import flwr as fl
from flwr.client import NumPyClient
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, Parameters, FitRes, Weights, parameters_to_weights, weights_to_parameters
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans
import numpy as np
import numpy.random as rand
import sys
sys.path.append('../')
import clustering.py.common_fn as my_fn
from sklearn.ensemble._hist_gradient_boosting import loss

class KFEDStrategy(FedAvg):
    
    def __init__(self,
                 seed: int = 51550,
                 fraction_fit: float = 0.1,
                 fraction_eval: float = 0.1,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn: Optional[
                     Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None):
        super().__init__(
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters)
        self.rng = np.random.default_rng(seed)
    
    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):#->Optional[Weights]:
        # get step
        config = self.on_fit_config_fn(rnd)
        # discriminate the aggregation to be performed
        if config['model'] == 'k-FED':
            # initial checks
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}
            # getting all centroids
            all_centroids = np.array([parameters_to_weights(fit_res.parameters) for _, fit_res in results])
            # pick, randomly, one client's centroids
            idx = self.rng.integers(0, all_centroids.shape[0], 1)
            # basis to be completed
            base_centroids = all_centroids[idx][0]
            # all other centroids
            other_centroids = all_centroids[np.arange(len(all_centroids))!=idx]
            other_centroids = np.concatenate(other_centroids, axis=0)
            # loop for completing the basis
            while base_centroids.shape[0] < config['n_clusters']:
                # all distances from the basis of centroids
                distances = [my_fn.distance_from_centroids(base_centroids, c) for c in other_centroids]
                # get the index of the maximum distance
                idx = np.argmax(distances)
                # add the new centroid
                base_centroids = np.concatenate((base_centroids, [other_centroids[idx]]), axis=0)
            print(base_centroids.shape)
            return weights_to_parameters(base_centroids), {}
        else :
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            return aggregated_weights