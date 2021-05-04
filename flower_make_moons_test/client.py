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
        'Give also the number of clients in the FL set up to build properly the dataset.\n' + \
        'One can optionally give the server location to pass the client builder.\n' + \
        'One can optionally give the number of local epochs to perform.\n' + \
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
                        help='client identifier')
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
                        help='number of local epochs to perform at each federated epoch')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
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
                        help='set true for producing a rotated dataset')
    parser.add_argument('--is_traslated',
                        dest='is_traslated',
                        required=False,
                        type=bool,
                        action='store',
                        help='set true for producing a traslated dataset')
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
                        help='tells the program whether to plot decision boundary, every 100 federated epochs, or not')
    parser.add_argument('--dump_curve',
                        dest='l_curve',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to dump the learning curve, at every federated epoch, or not')
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

    # managing parameters
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
        
    if not args.seed:
        SEED = 51550
    else:
        SEED = args.seed
        
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
        random.seed(SEED)
        TEST_RAND_STATE = random.randint(0, 100000)
        (x_test, y_test) = datasets.make_moons(n_samples=1000,
                                            shuffle=True,
                                            noise=0.1,
                                            random_state=TEST_RAND_STATE)

    if not args.server:
        SERVER = "192.168.1.191:5223"
    else:
        SERVER = args.server
    print("Server address: "+SERVER)

    # dataset, building the whole one and get the local
    x_tot, y_tot = my_fn.build_dataset(N_CLIENTS, N_SAMPLES, R_NOISE, SEED)  
    x_train, y_train = my_fn.get_client_dataset(args.client_id, N_CLIENTS, x_tot, y_tot) 
     
    if IS_ROT: 
        theta = math.pi/5 # (-1 + 2*random.random())*(math.pi/10)
        x_train = my_fn.rotate_moons(theta, x_train)
    if IS_TR: 
        dx = 0.2#*(-1 + 2*random.random())
        dy = 0.2#*(-1 + 2*random.random())
        x_train = my_fn.translate_moons(dx, dy, x_train)
    
    if not TEST :
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=5)
        train, test = next(kfold.split(x_train, y_train))
        x_test = x_train[test]
        y_test = y_train[test]
        x_train = x_train[train]
        y_train = y_train[train]
    
    # plotting the actual client dataset and saving the image to default folder
    my_fn.plot_client_dataset(args.client_id, x_train, y_train, x_test, y_test)

    class MakeMoonsClient(fl.client.NumPyClient):
        """Client object, to set client performed operations."""
        
        def __init__( self ):
            # initialize the number of federated epoch to zero
            self.f_round = 0
        
        def get_parameters(self):  # type: ignore
            """Get the model weights by model object."""
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            """Perform the fit step after having assigned new weights."""
            # increasing the number of epoch
            self.f_round += 1
            if self.f_round%10 == 0 : # some logging
                print("Federated Round number " + str(self.f_round))
            # getting weights from server
            model.set_weights(parameters)
            # performing the fit step
            model.fit(x_train, y_train, epochs=N_LOC_EPOCHS, verbose=0)#, batch_size=32)
            # returning the parameters necessary for FedAvg
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            """Perform the evaluation step after having assigned new weights."""
            # getting weights from server
            model.set_weights(parameters)
            # set verbosity level for evaluation step
            eval_verb = 0
            if self.f_round%100 == 0 and PLOT: # plotting decision boundary if requested
                eval_verb = 1 # increasing verbosity level
                my_fn.plot_decision_boundary(model, x_test, y_test, args.client_id)
            loss, accuracy = model.evaluate(x_test, y_test, verbose=eval_verb)
            # dmping learning curve record if requested
            if DUMP: my_fn.dump_learning_curve("l_curve_"+str(args.client_id), self.f_round, loss, accuracy)
            # returning the performance to the server to aggregate
            return loss, len(x_test), {"accuracy": accuracy}
    
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=MakeMoonsClient())
