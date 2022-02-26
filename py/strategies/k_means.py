#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from flwr.common import (FitRes, Parameters, Scalar, Weights, FitRes,
                         parameters_to_weights, weights_to_parameters)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from py.util import distance_from_centroids

class KMeansStrategy(FedAvg):

    def __init__(self,
                 out_dir: Union[Path, str] = None,
                 seed: int = 51550,
                 method: str = 'max_min',
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
        self.method = method
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
        all_centroids = []
        all_centroids_multi = []
        n_samples = []
        for _, fit_res in results:
            f_r = parameters_to_weights(fit_res.parameters)
            for i in range(config['n_clusters']):
                all_centroids.append(f_r[2*i])
                for _ in range(f_r[int(2*i+1)]):
                    all_centroids_multi.append(f_r[2*i])
                n_samples.append(f_r[int(2*i+1)])
        all_centroids = np.array(all_centroids)
        all_centroids_multi = np.array(all_centroids_multi)
        n_samples = np.array(n_samples)
        print('All centroids\' multi shape: {}'.format(all_centroids_multi.shape))
        print('All centroids\' shape: {}'.format(all_centroids.shape))
        print('N samples shape: {}'.format(n_samples.shape))
        pd.DataFrame(all_centroids_multi).to_csv(self.out_dir/'centroids_multi.csv')
        
        if self.method == 'double_kmeans':
            kmeans = KMeans(n_clusters=config['n_clusters'], n_init=20)
            predicted = kmeans.fit_predict(all_centroids_multi)
            base_centroids = kmeans.cluster_centers_
        
        if self.method == 'max_min':
            # pick, randomly, one client's first centroids
            idx = self.rng.integers(0, all_centroids.shape[0], 1)
            # basis to be completed
            base_centroids = np.array(all_centroids[idx])
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
        
        if self.method == 'random':            
            ## weight by n samples the centroids set!!!
            self.rng.shuffle(all_centroids)
            base_centroids = all_centroids[:config['n_clusters']]
        
        if self.method == 'random_weighted':            
            ## weight by n samples the centroids set!!!
            self.rng.shuffle(all_centroids_multi)
            base_centroids = all_centroids_multi[:config['n_clusters']]
        
        print('Basis centroids\' shape: {}'.format(base_centroids.shape))
        # Save base_centroids
        print(f"Saving base centroids...")
        np.savez(self.out_dir/'agg_clusters_centers.npz', *base_centroids)
        return weights_to_parameters(base_centroids), {}
