#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import os
import argparse
from argparse import RawTextHelpFormatter
import flwr as fl
# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def parse_args ():
    """Parse the arguments passed."""
    description = 'Server for moons test FL network using flower.\n' + \
        'Give the nuber of federated rounds to pass to the strategy builder.\n' + \
        'Give the minimum number of clients to wait for a federated averaging step.'

    parser = argparse.ArgumentParser(description = description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--rounds',
                        dest='rounds',
                        required=True,
                        type=int,
                        action='store',
                        help='number of federated rounds to perform')
    parser.add_argument('--n_clients',
                        dest='clients',
                        required=True,
                        type=int,
                        action='store',
                        help='minimum number of active clients to perform an iteration step')

    _args = parser.parse_args()

    return _args


# Start Flower server
if __name__ == "__main__":
    # parsing arguments
    args = parse_args()
    # instantiating the strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=args.clients
    )
    # starting the server
    fl.server.start_server("localhost:8081",
                           config={"num_rounds": args.rounds},
                           strategy=strategy)
