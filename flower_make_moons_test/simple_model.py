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
import flower_make_moons_test.common_fn as my_fn
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # w/o GPU

#%%

def parse_args():
    """Parse the arguments passed."""
    description = 'Simple model to compare results of FL approach.\n' + \
        'It simulates, once properly set, the same set up of a FL distribution in an aggregated version.\n' + \
        'The learning curve is dumped at every epoch.'
    parser = argparse.ArgumentParser(description = description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--n_clients',
                        dest='n_clients',
                        required=True,
                        type=int,
                        action='store',
                        help='maximum number of different clients to simulate, used to create the dataset')
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
                        help='number of total epochs for the training')
    parser.add_argument('--is_traslated',
                        dest='is_traslated',
                        required=False,
                        type=bool,
                        action='store',
                        help='set true in the case of a traslated datset')
    parser.add_argument('--is_rotated',
                        dest='is_rotated',
                        required=False,
                        type=bool,
                        action='store',
                        help='set true in the case of a rotated datset')
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
                        help='tells the program whether to plot decision boundary, every 100 epochs, or not')
    _args = parser.parse_args()
    return _args

#%%
if __name__ == "__main__":
    
    # parsing arguments
    args = parse_args()
    N_EPOCHS = args.n_epochs
    N_CLIENTS = args.n_clients
    N_SAMPLES = args.n_samples
    R_NOISE = 0.2
    
    if not args.plot:
        PLOT = True
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
    
    # building the dataset
    x, y = my_fn.build_dataset(n_clients=N_CLIENTS, noise=NOISE, total_samples=N_SAMPLES)#OLD, is_rotated=IS_ROT, is_translated=IS_TR)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=5)
    train, test = next(kfold.split(x, y))
    x_test = x[test]
    y_test = y[test]
    x_train = x[train]
    y_train = y[train]
    # plotting the dataset and save the figure in the default folder
    my_fn.plot_client_dataset('nofed', x_train, y_train, x_test, y_test)
    # creating and compiling the model
    model = my_fn.create_keras_model()
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    # running the training
    for i in range(N_EPOCHS):
        # fitting step
        model.fit(x_train, y_train, epochs=1, verbose=0)
        # evaluation step
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        # dumping a record of the learning curve to the default folder
        my_fn.dump_learning_curve('l_curve_nofed', i, loss, acc)
        # plotting, if requested, the decision boundary
        if PLOT and i%100==0: my_fn.plot_decision_boundary(model, x_test, y_test)
        