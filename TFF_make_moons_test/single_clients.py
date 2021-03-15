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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
# Place tensors on the CPU
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  
path = "/home/relogu/OneDrive/UNIBO/Magistrale/GNN_project/py/make_moons_test/CLIENTS_OUT/"
  
def printWeightsToFile(weights, filename):
    """Print in two files the weights of the simple keras model."""
    i=0
    fn=""
    for w in weights:
        fn=path+filename+str(i)+".txt"
        np.savetxt(fn, w)
        i+=1
        
def sendWeightsToServer(weights, filename):
    """Send the files containing the weights of the model to server."""
    i=0
    fn=""
    for w in weights:
        fn=path+filename+str(i)+".txt"
        with open(fn) as f:
            content=f.read()
            f.close()
        ret = 0
        fn = filename+str(i)+".txt"
        while ret < 200:
            response = requests.post("http://localhost:5000/api/v1/files/"+fn,
                                     data=content.encode('utf-8'))
            ret = response.status_code

        i+=1
        
def readWeightsFromFile(init_weights, filename):
    """Read from the file given the weights of the model."""
    for i in range(len(init_weights)):
        if i==2 : init_weights[i] = np.loadtxt(path+filename+str(i)+".txt", ndmin=2)
        elif i==3 : init_weights[i] = np.loadtxt(path+filename+str(i)+".txt", ndmin=1)
        else : init_weights[i] = np.loadtxt(path+filename+str(i)+".txt")
    return init_weights

def getServerWeights():
    """Retrieve server weights, from server."""
    for i in range(len(glob.glob(path+'ser*'))):
        response = requests.get('http://localhost:5000/api/v1/files/server_'+str(i)+'.txt')
        with open(path+'server_'+str(i)+'.txt', 'w') as f:
            f.write(response.text)
            f.close()

initializer = tf.keras.initializers.GlorotUniform()

def create_keras_model():
    """Define the model."""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(4, activation='tanh',
                              kernel_initializer=initializer,
                              bias_initializer='ones',),
        tf.keras.layers.Dense(2, activation='softmax',
                              kernel_initializer=initializer,
                              bias_initializer='ones',)])

def create_keras_null_model():
    """Define the model."""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(4, activation='tanh', kernel_initializer='zeros'),
        tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='zeros',
                              bias_initializer='zeros')])


with tf.device('/CPU:0'):
    
    # initializing null server weights to server
    null_weights = create_keras_null_model().weights
    printWeightsToFile(null_weights, "server_")
    sendWeightsToServer(null_weights, "server_")
    
    N_TRAIN = 20
    N_TEST = 1000
    N_LOC_EPOCHS = 100
    #N_FED_EPOCHS = 100
    N_CLIENTS = 20
    R_NOISE = 0.2
    np.random.seed(51550)
    (X_test, y_test) = datasets.make_moons(n_samples=N_TEST, noise=R_NOISE/2,
                                   shuffle=True, random_state=51550)
    
    def plot_decision_boundary(model, fed_iter):
        """Plot the decision boundary given the predictions of the model."""
        plt.figure(figsize=(18, 9))
        ax = plt.subplot(1, 1, 1)
        ax.set_title("Decision boundary for the test set at federated iteration "+str(fed_iter))
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
        ax = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        ax = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
        plt.show(block=False)
    
    model = create_keras_model()
    initial_weights = model.weights
    model.compile(optimizer='adam',
                  #loss='binary_crossentropy',
                  #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    acc = 0
    old_acc = 0
    old_loss = 1
    j = 0
    RANDOM_STATES = np.random.randint(0, 1000*N_CLIENTS, N_CLIENTS)
    while acc < 0.99: #federated loop
        j+=1
        print('Federated Loop '+str(j))
        getServerWeights()        
        model.set_weights(readWeightsFromFile(model.weights, "server_"))
        #print('Actual server weights: '+str(model.weights))
        loss, acc = model.evaluate(X_test, y_test, batch_size=(200), verbose=2)
        #if j%10==0 : 
        plot_decision_boundary(model, j)
        
        for i in range(N_CLIENTS):
            (X, y) = datasets.make_moons(n_samples=N_TRAIN,
                                   shuffle=True, noise=R_NOISE,
                                   random_state=RANDOM_STATES[i])
            '''
            figure = plt.figure(figsize=(18, 9))
            ax = plt.subplot(1, 1, 1)
            h = .02  # step size in the mesh
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            ax.set_title("Input data client "+str(i))
            ax.scatter(X[:N_TRAIN, 0], X[:N_TRAIN, 1], c=y[:N_TRAIN], edgecolors='k')
            # Plot the testing points
            ax.scatter(X[N_TRAIN:, 0], X[N_TRAIN:, 1], c=y[N_TRAIN:], alpha=0.6, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            '''
            if j == 1 or old_loss+1 < loss:
                model = create_keras_model()
                model.compile(optimizer='adam',
                              #loss='binary_crossentropy',
                              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
            
            #if j > 0:
                #model.reset_metrics()
                #print("Assigning server weights")
                #model.set_weights(readWeightsFromFile(model.weights, "server_"))
            
            #model.compile(optimizer='adam',
            #      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #      metrics=['accuracy'])
            
            model.fit(X, y, epochs=N_LOC_EPOCHS, verbose=0)
            
            printWeightsToFile(model.weights, "weights_c="+str(i)+"_")
            sendWeightsToServer(model.weights, "weights_c="+str(i)+"_")
            
        old_acc = acc
        old_loss = loss
        headers = {"n_clients": str(N_CLIENTS)}
        response = requests.get("http://localhost:5000/api/v1/launch_fed_avg", headers=headers)
        #print("Server averaging responded "+str(response.text))
        #print("with code "+str(response.status_code))


#%% testing something



    (X_test, y_test) = datasets.make_moons(n_samples=20000, noise=R_NOISE,
                                   shuffle=True, random_state=51550)

    model.fit(X, y, epochs=1000, verbose=1)
    plot_decision_boundary(model, j+1)
    i+=1



l = len(glob.glob(path+'ser*'))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu')])

pred = model(X).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
tf.keras.models.l
model.set_weights(m)
c = readWeightsForClient(1, model.weights, path+"weights_c=")

m = create_keras_null_model()
printWeightsToFile(m.weights, "null_weights_")
np.fromfile()
 
m=[.0]*4
for i in range(4):
    m[i] = np.loadtxt("null_weights_"+str(i)+".txt")
    
a = 0.18419312 + 0.83532268 + 0.46721321 - 0.75470078 + 0.55294448 + \
    - 0.79301023 + 0.2911692 - 0.46911666 - 0.6833601 + 0.39501935
a = a/10