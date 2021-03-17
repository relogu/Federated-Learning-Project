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
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import random
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
        'Give also the number of data samples you wanto to train for this client.\n' + \
        'One can optionally give the server location to pass the client builder.\n' + \
        'One can optionally give the noise to generate the dataset.\n' + \
        'One can optionally tell the program to plot the decision boundary at the evaluation step.\n' + \
        'The number of test data samples is fixed by the program.\n' + \
        'The client id will also initialize the seed for the train dataset.\n' + \
        'The program is built to make all the client use the same test dataset.'
    parser = argparse.ArgumentParser(description = description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--server',
                        dest='server',
                        required=False,
                        type=type(''),
                        action='store',
                        help='server address to point')
    parser.add_argument('--client_id',
                        dest='client_id',
                        required=True,
                        type=int,
                        action='store',
                        help='client id, set also the seed for the dataset')
    parser.add_argument('--rounds',
                        dest='rounds',
                        required=False,
                        type=int,
                        action='store',
                        help='number of local epochs to perform')
    parser.add_argument('--n_train',
                        dest='n_train',
                        required=True,
                        type=int,
                        action='store',
                        help='number of samples in training set')
    parser.add_argument('--noise',
                        dest='noise',
                        required=False,
                        type=float,
                        action='store',
                        help='noise to put in the train dataset')
    parser.add_argument('--plot',
                        dest='plot',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to plot decision boundary or not')
    _args = parser.parse_args()
    return _args


def plot_decision_boundary(model, fed_iter, x, y):
    """Plot the decision boundary given the predictions of the model."""
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Decision boundary for the test set at the federated round: " + str(fed_iter))
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.draw()
    plt.show(block=False)
    plt.savefig('output/dec_bound_F'+str(fed_iter)+'.png')
    plt.close()

if __name__ == "__main__":

    #parsing arguments
    args = parse_args()

    # Load and compile Keras model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(4, activation='tanh',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer='ones',),
        tf.keras.layers.Dense(2, activation='softmax',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              bias_initializer='ones',)])
    model.compile("adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # parameters
    N_TRAIN = int(args.n_train)
    if N_TRAIN < 10:
        N_TRAIN=10
    if not args.rounds:
        N_LOC_EPOCHS = 1
    else:
        N_LOC_EPOCHS = args.rounds
    if N_LOC_EPOCHS < 1:
        N_LOC_EPOCHS=1
    N_CLIENTS = 20
    if not args.noise:
        R_NOISE = 0.2
    else:
        R_NOISE = args.noise
    if not args.plot:
        PLOT = False
    else:
        PLOT = args.plot
    random.seed(51550)
    # TODO: control if these random states are equal and manage it
    TEST_RAND_STATE = random.randint(0, 100000)
    random.seed(args.client_id)
    TRAIN_RAND_STATE = random.randint(0, 100000)

    # datasets
    (x_train, y_train) = datasets.make_moons(n_samples=N_TRAIN,
                                             shuffle=True,
                                             noise=R_NOISE,
                                             random_state=TRAIN_RAND_STATE)
    (x_test, y_test) = datasets.make_moons(n_samples=1000,
                                           shuffle=True,
                                           noise=0.2,
                                           random_state=TEST_RAND_STATE)

    class MakeMoonsClient(fl.client.NumPyClient):
        """Client object, to set client performed operations."""
        
        def __init__( self ):
            self.f_round = 0
        
        def get_parameters(self):  # type: ignore
            """Get the model weights by model object."""
            print("Getting parameters step.")
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            """Perform the fit step after having assigned new weights."""
            print("Fit step.")
            #print(config)
            self.f_round += 1
            print("Federated Round number " + str(self.f_round))
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=N_LOC_EPOCHS, verbose=0)#, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            """Perform the evaluation step after having assigned new weights."""
            print("Evaluation step.")
            #print(config)
            model.set_weights(parameters)
            if self.f_round%100 == 0 and PLOT:
                loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
                plot_decision_boundary(model, self.f_round, x_test, y_test)
            else :
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            return loss, len(x_test), {"accuracy": accuracy}

    if not args.server:
        SERVER = "192.168.1.191:5223"
    else:
        SERVER = args.server
    print(SERVER)
    # Start Flower client
    fl.client.start_numpy_client(SERVER, client=MakeMoonsClient())
