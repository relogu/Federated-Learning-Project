#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:55:15 2021

@author: relogu
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from pathlib import Path
from functools import partial
import numpy as np
from flwr.common import (FitRes, Parameters, Scalar, Weights, FitRes,
                         parameters_to_weights)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class DECModelStrategy(FedAvg):
    
    def __init__(self,
                 out_dir: Union[Path, str] = None,
                 threshold: float = 0.001,
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
                     int, bool], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None):
        self.passed_on_fit_config_fn = on_fit_config_fn
        self.threshold = threshold
        super().__init__(
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            partial(on_fit_config_fn, True),
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters)
        if out_dir is None:
            self.out_dir = Path('')
        else:
            self.out_dir = Path(out_dir)
        print("Strategy output filename: {}".format(self.out_dir))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        config = self.on_fit_config_fn(rnd)
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if config['train']:
            # Retrieve metric and number of samples
            tols = [r.metrics["tol"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            # Aggregate and print custom metric
            tol_aggregated = sum(tols) / sum(examples)
            print("Aggregated tolerance is {}".format(tol_aggregated))
            if tol_aggregated < self.threshold:
                self.on_fit_config_fn = partial(self.passed_on_fit_config_fn, False)
                config = self.on_fit_config_fn(rnd)
                print("Stopping training")
                print("Saving aggregated_weights...")
                filename = "agg_weights_{}".format(config['model'])
                parameters = np.array(parameters_to_weights(
                    aggregated_weights[0]), dtype=object)
                np.savez(self.out_dir/filename, parameters)
        # aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # if aggregated_weights is not None and config['last']:
        #     print("Saving aggregated_weights...")
        #     filename = "agg_weights_{}".format(config['model'])
        #     parameters = np.array(parameters_to_weights(
        #         aggregated_weights[0]), dtype=object)
        #     np.savez(self.out_dir/filename, parameters)
        return aggregated_weights
