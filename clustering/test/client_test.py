#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:15 2021

@author: relogu
"""
import argparse
import os
import flwr as fl
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import KFold
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import random
import math
import pathlib
import sys
sys.path.append('../')
import flower_make_moons_test.common_fn as my_fn
path = pathlib.Path(__file__).parent.absolute()
path_to_test = str(path)
path_parent = str(path.parent)
sys.path.append(path_parent)
# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"

if __name__ == "__main__":

    # Load and compile Keras model
    model = my_fn.create_keras_model()
    model.compile("adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    # server address
    SERVER = "192.168.1.191:5223"
    
    class MakeMoonsClient(fl.client.NumPyClient):
        """Client object, to set client performed operations."""
        
        def __init__( self ):
            # initialize the number of federated epoch to zero
            self.f_round = 0
            model.load_weights(path_to_test+"/model.h5")
        
        def get_parameters(self):  # type: ignore
            """Get the model weights by model object."""
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            """Perform the fit step after having assigned new weights."""
            # increasing the number of epoch
            self.f_round += 1
            # getting weights from server
            model.set_weights(parameters)
            new_weights = model.get_weights()
            model.load_weights(path_to_test+"/model.h5")
            test_weights = model.get_weights()
            for t,n in zip(test_weights, new_weights):
                np.testing.assert_array_equal(t,n,'different weights')
            # returning the parameters necessary for FedAvg
            return model.get_weights(), 1, {}

        def evaluate(self, parameters, config):  # type: ignore
            """Perform the evaluation step after having assigned new weights."""
            # getting weights from server
            model.set_weights(parameters)
            # set verbosity level for evaluation step
            eval_verb = 0
            loss, accuracy = model.evaluate(x_test, y_test, verbose=eval_verb)
            # returning the performance to the server to aggregate
            return loss, len(x_test), {"accuracy": accuracy}
    
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=MakeMoonsClient())
