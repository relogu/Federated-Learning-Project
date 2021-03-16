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
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"

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
    parser.add_argument('--address',
                        dest='addres',
                        required=False,
                        type=type(''),
                        action='store',
                        help='complete address to launch server, e.g. 127.0.0.1:8081')

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

    if not args.server:
        SERVER = "192.168.1.191:5223"
    else:
        SERVER = args.server
    # starting the server
    fl.server.start_server(SERVER,
                           config={"num_rounds": args.rounds},
                           strategy=strategy)
