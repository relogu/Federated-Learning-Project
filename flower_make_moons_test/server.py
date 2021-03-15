#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import os
import argparse
import flwr as fl
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def parse_args ():
    """Parse the arguments given."""
    description = 'Client for moons test learning network using flower'

    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('--rounds',
                        dest='rounds',
                        required=True,
                        type=int,
                        action='store',
                        help='number of federated rounds to perform')

    _args = parser.parse_args()

    return _args


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    #parsing arguments
    args = parse_args()
    fl.server.start_server("localhost:8081", config={"num_rounds": args.rounds})
