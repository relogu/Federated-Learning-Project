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
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Data samples for the client " + str(client_id))
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
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.draw()
    #plt.show(block=False)
    plt.savefig('output/data_client_'+str(client_id)+'.png')
    plt.close()

def plot_decision_boundary(model, X_test, y_test):
    """Plot the decision boundary given the predictions of the model."""
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Fianl decision boundary for the test set")
    # Set min and max values and give it some padding
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    plt.show(block=False)
    
def translate_moons(dx, dy, X):
    """Translate using the vector (dx, dy) the make_moons dataset X."""
    if X.shape == (1,1):
        X[:, 0] = X[:, 0] + dx
        X[:, 1] = X[:, 1] + dy
    else :
        print("X has not the correct shape")
    return X
    
def rotate_moons(theta, X):
    """Rotate using the angle theta the make_moons dataset X."""
    if X.shape == (1,1):
        X[:, 0] = X[:, 0] + math.cos(theta)
        X[:, 1] = X[:, 1] + math.sin(theta)
    else :
        print("X has not the correct shape")
    return X