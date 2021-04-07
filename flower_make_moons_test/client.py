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
# disable possible gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# for debug connection
os.environ["GRPC_VERBOSITY"] = "debug"

def parse_args():
    """Parse the arguments passed."""
    description = 'Client for moons test FL network using flower.\n' + \
        'Give the id of the client and the number of local epoch you want to perform.\n' + \
        'Give also the number of data samples you want to train for this client.\n' + \
        'One can optionally give the server location to pass the client builder.\n' + \
        'One can optionally give the noise to generate the dataset.\n' + \
        'One can optionally tell the program to plot the decision boundary at the evaluation step.\n' + \
        'One can optionally tell the program to use the shared test set (default) or the train set as test also.\n' + \
        'The client id will also initialize the seed for the train dataset.\n'
    parser = argparse.ArgumentParser(description = description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--client_id',
                        dest='client_id',
                        required=True,
                        type=int,
                        action='store',
                        help='client id, set also the seed for the dataset')
    parser.add_argument('--n_samples',
                        dest='n_samples',
                        required=True,
                        type=int,
                        action='store',
                        help='number of total samples in whole training set')
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        action='store',
                        help='number of total clients in the FL setting')
    parser.add_argument('--server',
                        dest='server',
                        required=False,
                        type=type(''),
                        action='store',
                        help='server address to point')
    parser.add_argument('--rounds',
                        dest='rounds',
                        required=False,
                        type=int,
                        action='store',
                        help='number of local epochs to perform')
    parser.add_argument('--noise',
                        dest='noise',
                        required=False,
                        type=float,
                        action='store',
                        help='noise to put in the train dataset')
    parser.add_argument('--is_rotated',
                        dest='is_rotated',
                        required=False,
                        type=bool,
                        action='store',
                        help='for producing a rotated dataset')
    parser.add_argument('--is_traslated',
                        dest='is_traslated',
                        required=False,
                        type=bool,
                        action='store',
                        help='for producing a traslated dataset')
    parser.add_argument('--test',
                        dest='test',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to use the shared test dataset (True) or the train dataset as test (False)')
    parser.add_argument('--plot',
                        dest='plot',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to plot decision boundary or not')
    parser.add_argument('--dump_curve',
                        dest='l_curve',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to dump the learning curve or not')
    _args = parser.parse_args()
    return _args

if __name__ == "__main__":

    #parsing arguments
    args = parse_args()

    # Load and compile Keras model
    model = my_fn.create_keras_model()
    model.compile("adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # parameters
    N_SAMPLES = args.n_samples
    if N_SAMPLES < 10:
        N_SAMPLES = 10
        
    N_CLIENTS = args.n_clients
    if N_CLIENTS < 2:
        N_CLIENTS = 2
        
    if not args.rounds:
        N_LOC_EPOCHS = 1
    else:
        N_LOC_EPOCHS = args.rounds
    if N_LOC_EPOCHS < 1:
        N_LOC_EPOCHS = 1
        
    if not args.noise:
        R_NOISE = 0.1
    else:
        R_NOISE = args.noise
        
    if not args.is_rotated:
        IS_ROT = False
    else:
        IS_ROT = args.is_rotated
        
    if not args.is_traslated:
        IS_TR = False
    else:
        IS_TR = args.is_traslated
        
    if not args.plot:
        PLOT = True
    else:
        PLOT = args.plot
        
    if not args.l_curve:
        DUMP = True
    else:
        DUMP = args.l_curve
        
    if not args.test:
        TEST = False
    else:
        TEST = args.test
        
    random.seed(51550)
    # TODO: control if these random states are equal and manage it
    TEST_RAND_STATE = random.randint(0, 100000)
    random.seed(args.client_id)
    TRAIN_RAND_STATE = random.randint(0, 100000)

    # datasets    
    x_tot, y_tot = my_fn.build_dataset(8, N_SAMPLES, R_NOISE)  
    x_train, y_train = my_fn.get_client_dataset(args.client_id, N_CLIENTS, x_tot, y_tot) 
     
    if IS_ROT: 
        theta = (-1 + 2*random.random())*(math.pi/10)
        x_train = my_fn.rotate_moons(theta, x_train)
    if IS_TR: 
        dx = 0.2#*(-1 + 2*random.random())
        dy = 0.2#*(-1 + 2*random.random())
        x_train = my_fn.traslate_moons(dx, dy, x_train)
    
    if TEST :
        (x_test, y_test) = datasets.make_moons(n_samples=1000,
                                            shuffle=True,
                                            noise=0.1,
                                            random_state=TEST_RAND_STATE)
    else :
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=5)
        train, test = next(kfold.split(x_train, y_train))
        x_test = x_train[test]
        y_test = y_train[test]
        x_train = x_train[train]
        y_train = y_train[train]
        
    my_fn.plot_client_dataset(args.client_id, x_train, y_train, x_test, y_test)

    class MakeMoonsClient(fl.client.NumPyClient):
        """Client object, to set client performed operations."""
        
        def __init__( self ):
            self.f_round = 0
        
        def get_parameters(self):  # type: ignore
            """Get the model weights by model object."""
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            """Perform the fit step after having assigned new weights."""
            self.f_round += 1
            if self.f_round%10 == 0 :
                print("Federated Round number " + str(self.f_round))
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=N_LOC_EPOCHS, verbose=0)#, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            """Perform the evaluation step after having assigned new weights."""
            model.set_weights(parameters)
            if self.f_round%100 == 0 and PLOT:
                loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
                my_fn.plot_decision_boundary(model, x_test, y_test, args.client_id)
            else :
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            if DUMP: my_fn.dump_learning_curve("l_curve_"+str(args.client_id), self.f_round, loss, accuracy)
            return loss, len(x_test), {"accuracy": accuracy}

    if not args.server:
        SERVER = "192.168.1.191:5223"
    else:
        SERVER = args.server
    print(SERVER)
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=MakeMoonsClient())
