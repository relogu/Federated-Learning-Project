#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:25:15 2021

@author: relogu
"""
#%%
import tensorflow as tf
import argparse
from argparse import RawTextHelpFormatter
from sklearn import datasets
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import glob
import pathlib
import sys
import math
import random
sys.path.append('../')
import common_fn as my_fn
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%

def parse_args():
    """Parse the arguments passed."""
    description = 'Simple model to compare results of FL approach.'
    parser = argparse.ArgumentParser(description = description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        action='store',
                        help='number of different clients to simulate')
    parser.add_argument('--n_samples',
                        dest='n_samples',
                        required=True,
                        type=int,
                        action='store',
                        help='number of total samples')
    parser.add_argument('--n_epochs',
                        dest='n_epochs',
                        required=True,
                        type=int,
                        action='store',
                        help='number of epochs for the training')
    parser.add_argument('--is_traslated',
                        dest='is_traslated',
                        required=False,
                        type=bool,
                        action='store',
                        help='in the case of a traslated datset')
    parser.add_argument('--is_rotated',
                        dest='is_rotated',
                        required=False,
                        type=bool,
                        action='store',
                        help='in the case of a rotated datset')
    parser.add_argument('--noise',
                        dest='noise',
                        required=False,
                        type=float,
                        action='store',
                        help='noise to add to dataset')
    parser.add_argument('--plot',
                        dest='plot',
                        required=False,
                        type=bool,
                        action='store',
                        help='tells the program whether to plot decision boundary or not')
    _args = parser.parse_args()
    return _args

def build_dataset(n_clients, total_samples, noise,
                  is_translated=False, is_rotated=False):
    N_SAMPLES = total_samples/n_clients
    x=np.array(0)
    y=np.array(0)
    for i in range(n_clients):
        random.seed(i)
        train_rand_state = random.randint(0, 100000)
        (x_client, y_client) = datasets.make_moons(n_samples=int(N_SAMPLES), noise=noise,
                                shuffle=True, random_state=train_rand_state)
        if is_rotated: 
            theta = (-1 + 2*random.random())*(math.pi/10)
            x_client = my_fn.rotate_moons(theta, x_client)
        if is_translated: 
            dx = 0.2*(-1 + 2*random.random())
            dy = 0.2*(-1 + 2*random.random())
            x_client = my_fn.traslate_moons(dx, dy, x_client)
            
        if i == 0:
            x = x_client
            y = y_client
        else :
            x = np.concatenate((x, x_client), axis=0)
            y = np.concatenate((y, y_client), axis=0)       
    return x, y

#%%
if __name__ == "__main__":
    
    #parsing arguments
    args = parse_args()
    N_EPOCHS = args.n_epochs
    N_CLIENTS = args.n_clients
    N_SAMPLES = args.n_samples
    R_NOISE = 0.2 #TODO
    
    if not args.plot:
        PLOT = False
    else:
        PLOT = args.plot
        
    if not args.is_traslated:
        IS_TR = False
    else:
        IS_TR = args.is_traslated
        
    if not args.is_rotated:
        IS_ROT = False
    else:
        IS_ROT = args.is_rotated
        
    if not args.noise:
        NOISE = 0.1
    else:
        NOISE = args.noise
    
    x, y = build_dataset(n_clients=N_CLIENTS, noise=NOISE, total_samples=N_SAMPLES)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=5)
    train, test = next(kfold.split(x, y))
    x_test = x[test]
    y_test = y[test]
    x_train = x[train]
    y_train = y[train]

    my_fn.plot_client_dataset('nofed', x_train, y_train, x_test, y_test)

    model = my_fn.create_keras_model()
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    for i in range(N_EPOCHS):
        model.fit(x_train, y_train, epochs=1, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        my_fn.dump_learning_curve('nofed', i, loss, acc)
        if PLOT and i%100==0: my_fn.plot_decision_boundary(model, x_test, y_test)
        
