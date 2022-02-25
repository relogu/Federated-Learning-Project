#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

from pathlib import Path
import numpy as np
from flwr.common import (FitRes, Parameters, Scalar, Weights, FitRes,
                         parameters_to_weights, weights_to_parameters)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from py.util import distance_from_centroids

class KMeansStrategy(FedAvg):

    def __init__(self,
                 out_dir: Union[Path, str] = None,
                 seed: int = 51550,
                 fraction_fit: float = 0.1,
                 fraction_eval: float = 0.1,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn: Optional[
                     Callable[[Weights],
                              Optional[Tuple[float, Dict[str, Scalar]]]]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None) -> Optional[Weights]:
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
        if out_dir is None:
            self.out_dir = Path('')
        else:
            self.out_dir = Path(out_dir)
        print("Strategy output filename: {}".format(self.out_dir))
    
    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]):
        # get step
        config = self.on_fit_config_fn(rnd)
        # initial checks
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # getting all centroids --> (n_clients, n_centroids, n_dimensions)
        all_centroids = np.array([parameters_to_weights(
            fit_res.parameters) for _, fit_res in results])
        print('All centroids\' shape: {}'.format(all_centroids.shape))
        # all the centroids in one list
        all_centroids = all_centroids.reshape((all_centroids.shape[0]*all_centroids.shape[1], all_centroids.shape[2]))
        print('All centroids\' shape: {}'.format(all_centroids.shape))
        # pick, randomly, one client's first centroids
        idx = self.rng.integers(0, all_centroids.shape[0], 1)
        # basis to be completed
        base_centroids = np.array([all_centroids[idx]])
        print('Basis centroids\' starting shape: {}'.format(base_centroids.shape))
        # basis initial length
        basis_length = 1
        # loop for completing the basis
        while basis_length < config['n_clusters']:
            # all distances from the basis of centroids
            distances = [distance_from_centroids(
                base_centroids, c) for c in all_centroids]
            # get the index of the maximum distance
            idx = np.argmax(distances)
            # add the new centroid --> (n_centroids, n_dimensions)
            base_centroids = np.concatenate(
                (base_centroids, [all_centroids[idx]]), axis=0)
            basis_length = base_centroids.shape[0]
        # Save base_centroids
        print(f"Saving base centroids...")
        # with open(self.out_dir/'agg_clusters_centers.npz', 'w') as file:
        np.savez(self.out_dir/'agg_clusters_centers.npz', *base_centroids)
        return weights_to_parameters(base_centroids), {}
