#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:25:15 2021

@author: relogu
"""

import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import glob
import pathlib
import flower_make_moons_test.common_fn as my_fn
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

N_TRAIN = 20
N_TEST = 1000
N_EPOCHS = 100 # TODO: choose to be compatible with federated setting
N_CLIENTS = 20
R_NOISE = 0.2
np.random.seed(51550)

# TODO: build the dataset to be compatible with federated setting
(x1, y1) = datasets.make_moons(n_samples=N_TEST, noise=R_NOISE,
                               shuffle=True, random_state=51550)

my_fn.plot_client_dataset('nofed', x_train, y_train, _test, y_test)
model = my_fn.create_keras_model()
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
for i in range(N_EPOCHS):
    model.fit(X_train, y_test, epochs=N_EPOCHS, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    my_fn.dump_learning_curve('nofed', i, loss, acc)
    #if i%10==0: my_fn.plot_decision_boundary(model, X_test, y_test)
