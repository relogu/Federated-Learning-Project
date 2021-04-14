#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:25:15 2021

@author: relogu
"""

import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn import datasets

def create_keras_model():
    """Define the model."""
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(4, activation='tanh',
                              kernel_initializer=initializer,
                              bias_initializer='ones',),
        tf.keras.layers.Dense(2, activation='softmax',
                              kernel_initializer=initializer,
                              bias_initializer='ones',)])

def dump_learning_curve(filename, round, loss, accuracy):
    """Dump the learning curve."""
    path_to_file = pathlib.Path(__file__).parent.absolute()
    path_to_file = str(path_to_file)+"/output/"+filename+".dat"
    
    if round == 1: # first call, opening a new file
        file = open(path_to_file, "w")
        file.write("client,round,loss,accuracy\n")
    else :
        file = open(path_to_file, "a")
    file.write(filename+","+str(round)+","+str(loss)+","+str(accuracy)+"\n")
    file.close()

def plot_client_dataset(client_id, x_train, y_train, x_test, y_test):
    """Plot the data samples given the specified client id and dataset."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9))
    ax.set_title("Data samples for the client " + str(client_id))
    ax.set_xlabel('x')
    ax.set_ylabel('Y')
    '''
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    '''
    # Plot the samples
    plt.scatter(x_train[:, 0], x_train[:, 1],
                c=y_train, cmap=plt.cm.Spectral)
    y_test = y_test+2
    plt.scatter(x_test[:, 0], x_test[:, 1],
                c=y_test, cmap=plt.cm.Spectral)
    plt.draw()
    #plt.show(block=False)
    plt.savefig('output/data_client_'+str(client_id)+'.png')
    plt.close()

def plot_decision_boundary(model, x_test, y_test, client_id=None, fed_iter=None):
    """Plot the decision boundary given the predictions of the model."""
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    if fed_iter is None and client_id is None: ax.set_title("Final decision boundary for the test set")
    else: ax.set_title("Decision boundary for the test set at the federated round: " + str(fed_iter))
    if client_id is None: title = 'Decison boundary for aggregated model'
    else: title = 'Decison boundary for client-'+str(client_id)+' model'
    if fed_iter is not None: title += ' at iteration '+str(fed_iter)  
    ax.set_title(title)
    # Set min and max values and give it some padding
    x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
    y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    plt.draw()
    #plt.show(block=False)
    if client_id is None: filename = 'output/dec_bound_nofed'
    else: filename = 'output/dec_bound_c'+str(client_id)
    if fed_iter is None: filename += '.png'
    else: filename += '_e'+str(fed_iter)+'.png'    
    plt.savefig(filename)
    plt.close()
    
def traslate_moons(dx, dy, x):
    """Translate using the vector (dx, dy) the make_moons dataset x."""
    if x.shape[1] == 2:
        x[:, 0] = x[:, 0] + dx
        x[:, 1] = x[:, 1] + dy
    else :
        print("x has not the correct shape")
    return x
    
def rotate_moons(theta, x):
    """Rotate using the angle theta the make_moons dataset x."""
    xc = x.copy()
    if x.shape[1] == 2:
        xc[:, 0] = x[:, 0]*math.cos(theta) - x[:, 1]*math.sin(theta)
        xc[:, 1] = x[:, 0]*math.sin(theta) + x[:, 1]*math.cos(theta)
    else :
        print("x has not the correct shape")
    return xc

def build_dataset(n_clients, total_samples, noise,
                  is_translated=False, is_rotated=False):
    """Build the dataset with the specified features."""
    N_SAMPLES = int(total_samples/n_clients)
    x=np.array(0)
    y=np.array(0)
    for i in range(n_clients):
        random.seed(i) # maybe random.seed(51550)
        train_rand_state = random.randint(0, 100000)
        (x_client, y_client) = datasets.make_moons(n_samples=int(N_SAMPLES),
                                                   noise=noise,
                                                   shuffle=True,
                                                   random_state=train_rand_state)
        if is_rotated: 
            theta = (-1 + 2*random.random())*(math.pi/10)
            x_client = rotate_moons(theta, x_client)
        if is_translated: 
            dx = 0.2*(-1 + 2*random.random())
            dy = 0.2*(-1 + 2*random.random())
            x_client = traslate_moons(dx, dy, x_client)
            
        if i == 0:
            x = x_client
            y = y_client
        else :
            x = np.concatenate((x, x_client), axis=0)
            y = np.concatenate((y, y_client), axis=0)       
    return x, y

def get_client_dataset(client_id, n_clients, x_tot, y_tot):
    """Get the single client dataset given the whole dataset."""
    if len(x_tot.shape) == 2 and x_tot.shape[1] == 2 \
        and len(y_tot.shape) == 1 and y_tot.shape[0] == x_tot.shape[0]:
        n_samples = x_tot.shape[0]
        n_sam_client = int(n_samples/n_clients)
        for i in range(n_clients):
            if i != client_id:
                continue
            else:
                return x_tot[i*n_sam_client:(i+1)*n_sam_client], y_tot[i*n_sam_client:(i+1)*n_sam_client]
    else:
        print("parameters have not the correct shape")
        return [], []

