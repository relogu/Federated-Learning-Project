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
                         parameters_to_weights)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedYogi

class SaveModelStrategyFedAvg(FedAvg):
    
    def __init__(self,
                 out_dir: Union[Path, str] = None,
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
        if aggregated_weights is not None and config['last']:
            print("Saving aggregated_weights...")
            filename = "agg_weights_{}".format(config['model'])
            parameters = np.array(parameters_to_weights(
                aggregated_weights[0]), dtype=object)
            np.savez(self.out_dir/filename, parameters)
        return aggregated_weights

class SaveModelStrategyFedYogi(FedYogi):

    def __init__(
        self,
        out_dir: Union[Path, str] = None,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[
            int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ):
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
            initial_parameters,
            eta,
            eta_l,
            beta_1,
            beta_2,
            tau,)
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
        if aggregated_weights is not None and config['last']:
            print("Saving aggregated_weights...")
            filename = "agg_weights_{}".format(config['model'])
            parameters = np.array(parameters_to_weights(
                aggregated_weights[0]), dtype=object)
            np.savez(self.out_dir/filename, parameters)
        return aggregated_weights


class SaveModelStrategyFedAdam(FedAdam):
    
    def __init__(
        self,
        out_dir: Union[Path, str] = None,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights],
                     Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[
            int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    ):
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
            initial_parameters,
            eta,
            eta_l,
            beta_1,
            beta_2,
            tau,)
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
        if aggregated_weights is not None and config['last']:
            print("Saving aggregated_weights...")
            filename = "agg_weights_{}".format(config['model'])
            parameters = np.array(parameters_to_weights(
                aggregated_weights[0]), dtype=object)
            np.savez(self.out_dir/filename, parameters)
        return aggregated_weights

