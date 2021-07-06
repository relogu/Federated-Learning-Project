#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:11:01 2021

@author: relogu
"""

import math
import pathlib
import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from sklearn import datasets
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist


class PrepareData(Dataset):

    def __init__(self, x, y):
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_euromds_dataset(accept_nan: int = 0,
                        groups: list[str] = None,
                        exclude_cols: list[str] = None,
                        path_to_data: Union[Path, str] = None):
    # set the path
    if path_to_data is None:
        parent = pathlib.Path(__file__).parent.parent.absolute()
        data_folder = parent/'data'/'euromds'
    else:
        data_folder = path_to_data
    # get the groups dataframe
    df_groups = pd.read_csv(data_folder/'dataFrameGroups.csv')
    # get the groups of columns/variables
    unique_groups = pd.unique(df_groups['group'])
    # check for the selected groups
    if groups != None and not set(groups).issubset(set(unique_groups)):
        raise ValueError(
            'The list \'groups\' contains some values that are not in the groups\' dataframe')
    # get the selected columns
    if groups is None:
        selected_cols = df_groups['data']
    else:
        selected_cols = df_groups[df_groups['group'].isin(set(groups))]['data']
    # check for the excluded columns
    if exclude_cols != None:
        if not set(exclude_cols).issubset(set(selected_cols)):
            warnings.warn('The list \'exclude_cols\' contains some values that are not in the selected groups\' columns.\n'+\
                'These values will be removed from the list.',
                        Warning)
        selected_cols = list(set(selected_cols)-set(exclude_cols))
    # get the main dataframe
    main_df = pd.read_csv(data_folder/'dataFrame.csv')
    # select the columns
    main_df = main_df[selected_cols]
    # filtering nans
    filtered = main_df.copy()
    for c in main_df.columns:
        a = len(main_df[main_df[c].isnull()])
        if a > accept_nan:
            filtered = filtered.drop(columns=c)
    del main_df
    del groups
    del df_groups
    return filtered


def dump_labels_euromds(labels,
                        name: str = None,
                        path_to_data: Union[Path, str] = None):
    # set the path
    if path_to_data is None:
        parent = pathlib.Path(__file__).parent.parent.absolute()
        data_folder = parent/'data'/'euromds'
    else:
        data_folder = path_to_data
    # set the filename
    if name is None:
        filename = 'labels.csv'
    else:
        filename = 'labels_'+name+'.csv'
    # get the main dataframe
    main_df = pd.read_csv(data_folder/'dataFrame.csv')
    # selected the IDs
    main_df = main_df[main_df.columns[0]]
    # building the dataframe
    main_df = pd.DataFrame({'ID': main_df,
                            'label': labels})
    # saving the file
    main_df.to_csv(data_folder/filename)


def get_outcome_euromds_dataset(accept_nan: int = 0,
                                groups: list[str] = None,
                                exclude_cols: list[str] = None,
                                path_to_data: Union[Path, str] = None):
    # set the path
    if path_to_data is None:
        parent = pathlib.Path(__file__).parent.parent.absolute()
        data_folder = parent/'data'/'euromds'
    else:
        data_folder = path_to_data
    # get the groups dataframe
    raw_cat = pd.read_csv(data_folder/'dataRawCategories.csv')
    # get the groups of columns/variables
    out_cat = raw_cat[raw_cat['category']=='Outcome Data']
    # get columns' names
    cols = out_cat['data']
    # raw df
    raw_df = pd.read_csv(data_folder/'dataRaw.csv')
    # get outcomes df
    prev = raw_df[cols]
    # filtering
    prev = prev.replace('no', 1)
    prev = prev.replace('yes', 0)
    prev = prev.replace('Alive', 1)
    prev = prev.replace('Dead', 0)
    ret = {}
    i = 0
    for col in prev.columns:
        ret['outcome_{}'.format(i)] = prev[col]
        i += 1
    return pd.DataFrame(ret)
    


def split_dataset(x,
                  y=None,
                  splits: int = 5,
                  fold_n: int = 0,
                  shuffle: bool = False,
                  r_state: int = 51550):
    if fold_n < 0 or fold_n > splits-1:
        raise ValueError(
            'The fold number, fold_n, cannot be lower than zero or higher than the number of splits minus one, splits-1')
    # Define the K-fold Cross Validator
    if shuffle:
        kfold = KFold(n_splits=splits,
                      shuffle=shuffle,
                      random_state=r_state)
    else:
        kfold = KFold(n_splits=splits)
    if y is None:
        train, test = next(kfold.split(x))
        x_test = x[test].copy()
        x_train = x[train].copy()
        return x_train, x_test
    else:
        train, test = next(kfold.split(x, y))
        x_test = x[test].copy()
        y_test = y[test].copy()
        x_train = x[train].copy()
        y_train = y[train].copy()
        return x_train, y_train, x_test, y_test


def translate_2d(dx: float, dy: float, x):
    """Translate using the vector (dx, dy) the make_moons dataset x.
    The function will retrieve a copy of x.

    Args:
        dx (float): x-component of the translation vector
        dy (float): y-component of the translation vector
        x (ndarray of shape (n_samples, 2)): list of 2-D points generated by sklearn.datasets.make_moons()

    Returns:
        (ndarray of shape (n_samples, 2)): translated list of 2-D points generated by sklearn.datasets.make_moons()
    """
    # get a copy
    xc = x.copy()
    # check on shape
    if x.shape[1] == 2:
        # applying transformation
        xc[:, 0] = x[:, 0] + dx
        xc[:, 1] = x[:, 1] + dy
    else:
        # error msg
        raise TypeError("the input x has not the correct shape")
    return xc


def rotate_2d(theta: float, x):
    """Rotate using the angle theta the make_moons dataset x w.r.t the origin (0,0).
    The function will retrieve a copy of x.

    Args:
        theta (float): angle generator for the rotation transformation
        x (ndarray of shape (n_samples, 2)): list of 2-D points generated by sklearn.datasets.make_moons()

    Returns:
        (ndarray of shape (n_samples, 2)): rotated list of 2-D points generated by sklearn.datasets.make_moons()
    """
    # get a copy
    xc = x.copy()
    # check on shape
    if xc.shape[1] == 2:
        # applying tranformation
        xc[:, 0] = x[:, 0]*math.cos(theta) - x[:, 1]*math.sin(theta)
        xc[:, 1] = x[:, 0]*math.sin(theta) + x[:, 1]*math.cos(theta)
    else:
        # error msg
        raise TypeError("the input x has not the correct shape")
    return xc


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


def build_dataset(n_clients: int, total_samples: int, noise: float, seed: int = 51550):
    """Build the entire dataset, to be distributed.

    Args:
        n_clients (int): number of clients onto which distribute the whole dataset
        total_samples (int): total number of sample of the whole datset
        noise (float): the amount of noise to generate the dataset
        seed (int): the seed for the generator of the dataset

    Returns:
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels
    """
    # getting the number of samples of the clients' dataset
    N_SAMPLES = int(total_samples/n_clients)
    # initializing arrays of points and labels (may be not needed)
    x = np.array(0)
    y = np.array(0)
    # set the intial seed for the RN generator
    rng = np.random.default_rng(seed)
    # loop on clients
    for i in range(n_clients):
        # get a RN for the state of the dataset generator
        train_rand_state = rng.integer(0, 100000)
        # get data points and labels
        (x_client, y_client) = datasets.make_moons(n_samples=int(N_SAMPLES),
                                                   noise=noise,
                                                   shuffle=True,
                                                   random_state=train_rand_state)
        # fill the arrays of points and labels
        if i == 0:
            x = x_client
            y = y_client
        else:
            x = np.concatenate((x, x_client), axis=0)
            y = np.concatenate((y, y_client), axis=0)
    return x, y


def build_mnist_dataset(n_clients: int, total_samples: int, noise: float, seed: int = 51550):
    """Build the entire dataset, to be distributed.

    Args:
        n_clients (int): number of clients onto which distribute the whole dataset
        total_samples (int): total number of sample of the whole datset
        noise (float): the amount of noise to generate the dataset
        seed (int): the seed for the generator of the dataset

    Returns:
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels
    """
    # getting the number of samples of the clients' dataset
    N_SAMPLES = int(total_samples/n_clients)
    # initializing arrays of points and labels (may be not needed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x[0:N_SAMPLES], y[0:N_SAMPLES]


def get_client_dataset(client_id: int, n_clients: int, x_tot, y_tot):
    """Get the single client dataset given the whole dataset.

    Args:
        client_id (int): identifier of the client, must be inside [0, (n_clients) - 1]
        n_clients (int): number of clients onto which the whole dataset is being distributed
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels

    Returns:
        x (ndarray of shape (single_client_samples, 2)): vector of 2-D points relative to the client
        y (ndarray of shape (single_client_samples)): vector of numerical labels relative to the client
    """
    # check on the shapes of the inputs
    if client_id >= n_clients or client_id < 0:
        msg = "the input client_id has not an allowed value, " + \
            "insert a positive value lesser than n_clients"
        raise TypeError(msg)
    if len(x_tot.shape) != 2 or x_tot.shape[1] != 2:
        msg = "the input x_tot has not the correct shape"
        raise TypeError(msg)
    if len(y_tot.shape) != 1:
        msg = "the input y_tot has not the correct shape"
        raise TypeError(msg)
    if y_tot.shape[0] != x_tot.shape[0]:
        msg = "the inputs x_tot and y_tot have not compatible shapes, " + \
            "they must represent the same number of points"
        raise TypeError(msg)
    # get the total number of samples
    n_samples = x_tot.shape[0]
    # get the number of samples for the single clients
    n_sam_client = int(n_samples/n_clients)
    # loop on clients
    for i in range(n_clients):
        # continue on wrong clients and returning the right dataset
        if i != client_id:
            continue
        else:
            return x_tot[i*n_sam_client:(i+1)*n_sam_client].copy(), y_tot[i*n_sam_client:(i+1)*n_sam_client].copy()


def get_client_mnist_dataset(client_id: int, n_clients: int, x_tot, y_tot):
    """Get the single client dataset given the whole dataset.

    Args:
        client_id (int): identifier of the client, must be inside [0, (n_clients) - 1]
        n_clients (int): number of clients onto which the whole dataset is being distributed
        x (ndarray of shape (total_samples, 2)): vector of 2-D points
        y (ndarray of shape (total_samples)): vector of numerical labels

    Returns:
        x (ndarray of shape (single_client_samples, 2)): vector of 2-D points relative to the client
        y (ndarray of shape (single_client_samples)): vector of numerical labels relative to the client
    """
    # check on the shapes of the inputs
    if client_id >= n_clients or client_id < 0:
        msg = "the input client_id has not an allowed value, " + \
            "insert a positive value lesser than n_clients"
        raise TypeError(msg)
    if len(x_tot.shape) != 2 or x_tot.shape[1] != 784:
        msg = "the input x_tot has not the correct shape"
        raise TypeError(msg)
    if len(y_tot.shape) != 1:
        msg = "the input y_tot has not the correct shape"
        raise TypeError(msg)
    if y_tot.shape[0] != x_tot.shape[0]:
        msg = "the inputs x_tot and y_tot have not compatible shapes, " + \
            "they must represent the same number of points"
        raise TypeError(msg)
    # get the total number of samples
    n_samples = x_tot.shape[0]
    # get the number of samples for the single clients
    n_sam_client = int(n_samples/n_clients)
    # loop on clients
    for i in range(n_clients):
        # continue on wrong clients and returning the right dataset
        if i != client_id:
            continue
        else:
            return x_tot[i*n_sam_client:(i+1)*n_sam_client], y_tot[i*n_sam_client:(i+1)*n_sam_client]
