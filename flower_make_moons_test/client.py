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
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def parse_args ():
    """Parse the arguments given."""
    description = 'Client for moons test learning network using flower'

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('--client',
                        dest='client',
                        required=True,
                        type=int,
                        action='store',
                        help='client id, 0-19 allowed')

    parser.add_argument('--rounds',
                        dest='rounds',
                        required=True,
                        type=int,
                        action='store',
                        help='number of local rounds to perform')

    _args = parser.parse_args()

    return _args

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
    N_TRAIN = 20
    N_TEST = 1000
    N_LOC_EPOCHS = args.rounds
    if N_LOC_EPOCHS < 1:
        N_LOC_EPOCHS=1
    N_CLIENTS = 20
    R_NOISE = 0.2
    np.random.seed(51550)
    i=args.client
    #i=1
    if i<0:
        i=-i
    if i>19:
        i=19
    RANDOM_STATES = np.random.randint(0, 1000*N_CLIENTS, N_CLIENTS+1)

    # datasets
    (x_train, y_train) = datasets.make_moons(n_samples=N_TRAIN,
                                             shuffle=True,
                                             noise=R_NOISE,
                                             random_state=RANDOM_STATES[i])

    (x_test, y_test) = datasets.make_moons(n_samples=N_TEST,
                                           shuffle=True,
                                           noise=R_NOISE,
                                           random_state=RANDOM_STATES[-1])

    class MakeMoonsClient(fl.client.NumPyClient):
        """Client object, to set client performed operations."""

        def get_parameters(self):  # type: ignore
            """Get the model weights by model object."""
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            """Perform the fit step after having assigned new weights."""
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=N_LOC_EPOCHS)#, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            """Perform the evaluation step after having assigned new weights."""
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8081", client=MakeMoonsClient())
