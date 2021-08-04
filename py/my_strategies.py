#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from pathlib import Path
import flwr as fl
import numpy as np
import numpy.random as rand
from flwr.common import (FitRes, Parameters, Scalar, Weights, FitRes,
                         parameters_to_weights, weights_to_parameters)
from flwr.server.client_proxy import ClientProxy, EvaluateRes
from flwr.server.strategy import FedAvg
from py.util import distance_from_centroids

agg_weights_filename = "aggregated_weights.npz"


class SaveModelStrategy(FedAvg):

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
        if out_dir is None:
            self.out_dir = agg_weights_filename
        else:
            self.out_dir = Path(out_dir)/agg_weights_filename
        print("Strategy output filename: {}".format(self.out_dir))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):  # -> Optional[Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print("Saving aggregated_weights...")
            parameters = np.array(parameters_to_weights(
                aggregated_weights[0]), dtype=object)
            np.savez(self.out_dir, parameters)
        return aggregated_weights


class AggregateCustomMetricStrategy(FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):  # -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] *
                      r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


class MyStrategy(FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):  # -> Optional[Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print("Saving aggregated weights...")
            np.savez(agg_weights_filename, *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):  # -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] *
                      r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


class KFEDStrategy(FedAvg):

    def __init__(self,
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

    # ->Optional[Weights]:
    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
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
            # getting all centroids --> (n_clients, n_centroids, n_dimensions)
            all_centroids = np.array([parameters_to_weights(
                fit_res.parameters) for _, fit_res in results])
            print('All centroids\' shape: {}'.format(all_centroids.shape))
            # pick, randomly, one client's centroids
            idx = self.rng.integers(0, all_centroids.shape[0], 1)
            # basis to be completed
            base_centroids = all_centroids[idx][0]
            # all other centroids
            other_centroids = all_centroids[np.arange(
                len(all_centroids)) != idx]
            other_centroids = np.concatenate(other_centroids, axis=0)
            # loop for completing the basis
            while base_centroids.shape[0] < config['n_clusters']:
                # all distances from the basis of centroids
                distances = [distance_from_centroids(
                    base_centroids, c) for c in other_centroids]
                # get the index of the maximum distance
                idx = np.argmax(distances)
                # add the new centroid --> (n_centroids, n_dimensions)
                base_centroids = np.concatenate(
                    (base_centroids, [other_centroids[idx]]), axis=0)
                print(base_centroids.shape)
            # Save base_centroids
            print(f"Saving base centroids...")
            np.savez("base_centroids.npz", *base_centroids)
            return weights_to_parameters(base_centroids), {}
        else:
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            # Save aggregated_weights
            print("Saving aggregated weights...")
            np.savez(agg_weights_filename, *aggregated_weights)
            return aggregated_weights
