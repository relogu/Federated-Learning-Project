#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from lifelines import KaplanMeierFitter
from pathlib import Path
from typing import Union, Dict


def plot_points_2d(x, y):
    """Plot the points x coloring them by the labels in vector y

    Args:
        x (ndarray of shape (n_samples, 2)): vector of 2-D points to plot
        y (ndarray of shape (n_samples)): vector of numerical labels

    Returns:
        (matplotlib.pyplot.PathCollection)
    """
    return plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)


def plot_client_dataset_2d(client_id, x_train, y_train, x_test, y_test, path=None):
    """Plot and dump to a file the data samples given the specified client id and dataset.

    Args:
        client_id (str or int or cast to str): identifier for the client
        x_train (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the train set
        y_train (ndarray of shape (n_samples)): vector of numerical labels for the train set
        x_test (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the test set
        y_test (ndarray of shape (n_samples)): vector of numerical labels for the test set
    """
    # setting path for saving image
    if path is None:
        path = 'output'
    # initialize graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))
    ax.set_title("Data samples for the client " + str(client_id))
    ax.set_xlabel('x')
    ax.set_ylabel('Y')
    # Plot the samples
    plot_points_2d(x_train, y_train)
    # augment test to be colored differently
    y_test = y_test+2
    plot_points_2d(x_test, y_test)
    plt.draw()
    # plt.show(block=False)
    plt.savefig(path+'/data_client_'+str(client_id)+'.png')
    plt.close()

# TODO: correct warning for complexity


def plot_lifelines_pred(time,
                        event,
                        labels,
                        fed_iter=None,
                        client_id=None,
                        path_to_out: Union[Path, str] = None):
    # setting path for saving image
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(path_to_out)
    # initialize graph
    fig, axes = plt.subplots(1, 1, figsize=(25, 15))
    # setting title and filename
    if fed_iter is None and client_id is None:
        fig.suptitle("Final lifelines for the test set", fontsize=16)
        filename = 'lifelines_pred.png'
    elif fed_iter is None and client_id is not None:
        fig.suptitle("Actual lifelines for the test set for client {}".
                     format(client_id), fontsize=16)
        filename = 'lifelines_pred_'+str(client_id)+'.png'
    elif fed_iter is not None and client_id is None:
        fig.suptitle(
            "Lifelines for the test set at the federated round {}".
            format(str(fed_iter)), fontsize=16)
        filename = 'lifelines_pred_e'+str(fed_iter)+'.png'
    else:
        fig.suptitle(
            "Lifelines for the test set at the federated round {} for client {}".
            format(str(fed_iter), client_id), fontsize=16)
        filename = 'lifelines_pred_'+str(client_id)+'_e'+str(fed_iter)+'.png'
    # selected fitters
    fitters = {'KaplanMeierFitter': KaplanMeierFitter(),
               # 'WeibullFitter': WeibullFitter(),
               # 'ExponentialFitter': ExponentialFitter(),
               # 'LogNormalFitter': LogNormalFitter(),
               # 'LogLogisticFitter': LogLogisticFitter(),
               # 'PiecewiseExponentialFitter': PiecewiseExponentialFitter([40, 60]),
               # 'GeneralizedGammaFitter': GeneralizedGammaFitter()
               # 'SplineFitter': SplineFitter(T.loc[E.astype(bool)], [0, 50, 100])
               }
    # loop on fitters
    i = j = 0
    for key in fitters:
        ax = axes  # [i][j]
        # loop on labels
        for label in np.unique(labels):
            idx = (labels == label)
            unique, counts = np.unique(idx, return_counts=True)
            #print('Label {} has {} samples'.format(label, counts[unique==True]))
            if counts[unique == True] > 5:
                fitters[key].fit(time[idx], event[idx],
                                 label='f_{} l_{}'.format(key, label))
                fitters[key].plot_survival_function(ax=ax)
        i += 1
        if i > 1:
            i = 0
            j += 1
    # plt.show(block=False)
    # dump to a file
    plt.savefig(path_to_out/filename)
    plt.close()


def plot_dec_bound(model, x):
    """Plot the decision boundaries given by model.
    The vector x is used only to set the range of the axis.

    Args:
        model (tensorflow.keras.Model): model from which get the predictions
        x (ndarray of shape (n_samples, 2)): vector of 2-D points to plot

    Returns:
        (matplotlib.pyplot.QuadContourSet)
    """
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    return plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)


def plot_decision_boundary(model, x_test, y_test, client_id=None, fed_iter=None, path=None):
    """Plot the decision boundary given the predictions of the model.

    Args:
        model (tensorflow.keras.Model): model from which get the predictions
        x_test (ndarray of shape (n_samples, 2)): vector of 2-D points to plot for the train set
        y_test (ndarray of shape (n_samples)): vector of numerical labels for the train set
        client_id (str or int or cast to str, optional): identifier for the client. Defaults to None.
        fed_iter (int, optional): current federated step of building title. Defaults to None.
        path (str, optional): complete or relative path to output folder. Defaults to None.
    """
    # setting path for saving image
    if path is None:
        path = 'output'
    # initialize graph
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    if fed_iter is None and client_id is None:
        ax.set_title("Final decision boundary for the test set")
    else:
        ax.set_title(
            "Decision boundary for the test set at the federated round: " + str(fed_iter))
    if client_id is None:
        title = 'Decison boundary for aggregated model'
    else:
        title = 'Decison boundary for client-'+str(client_id)+' model'
    if fed_iter is not None:
        title += ' at iteration '+str(fed_iter)
    ax.set_title(title)
    # plot dec boundary
    plot_dec_bound(model, x_test)
    # plot test points
    plot_points_2d(x_test, y_test)
    plt.draw()
    # plt.show(block=False)
    # dump to a file
    if client_id is None:
        filename = path+'/dec_bound_nofed'
    else:
        filename = path+'/dec_bound_c'+str(client_id)
    if fed_iter is None:
        filename += '.png'
    else:
        filename += '_e'+str(fed_iter)+'.png'
    plt.savefig(filename)
    plt.close()


def print_confusion_matrix(y,
                           y_pred,
                           client_id=None,
                           fed_iter=None,
                           path_to_out: Union[Path, str] = None):
    # setting path for saving image
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(path_to_out)
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    # dump to a file
    if client_id is None:
        filename = 'conf_matrix_nofed'
    else:
        filename = ('conf_matrix_c'+str(client_id))
    if fed_iter is None:
        filename += '.png'
    else:
        filename += '_e'+str(fed_iter)+'.png'
    plt.savefig(path_to_out/filename)
    plt.close()
