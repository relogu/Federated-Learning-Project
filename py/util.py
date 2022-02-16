#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:11:01 2021

@author: relogu
"""

import numpy as np
import torch
from scipy.stats import skewnorm, uniform


# TODO: give by argument the distance function
# TODO: give by argument the closeness criteria
def distance_from_centroids(centroids_array, vector):
    # distances array
    distances = []
    # loop on the array/list of centroids
    for centroid in centroids_array:
        # distance between the current centroid and the given vector
        d = np.linalg.norm(centroid-vector)
        # append the distance
        distances = np.append(distances, d)
    # returning the lower distance between all the centroids
    return min(distances)


# TODO: give by argument the distance function
# TODO: give by argument the closeness criteria
def sync_centroids(server_centroids, client_centroids, metric=np.linalg.norm, verbose: bool = False):
    # labels array where index == server_label and
    # value == client_label
    labels = []
    # distances from the closest centroid identified
    distances = []
    # loop on the centroids of the server
    for centroid in server_centroids:
        # getting all the distances between the current centroid
        # and all the client centroids
        d = [metric(centroid, c) for c in client_centroids]
        if verbose:
            print('Distances from current server centroid and all client\'s centroids:\n\t{}'.
                  format(d))
        # getting the index of the minimum distances (closest centroid)
        client_label = np.argmin(d)
        # appending the read label to make the correspondance array
        labels = np.append(labels, client_label)
        # appending the correspondend distance
        distances = np.append(distances, d[client_label])
    return labels, distances


def compute_centroid_np(array):
    n_dim = len(array[0])
    length = array.shape[0]
    centroid = []
    for i in range(n_dim):
        sum_i = np.sum(array[:, i]) / length
        centroid = np.append(centroid, sum_i)
    return centroid


def complete_basis_centroids(base_centroids, other_centroids, n_clusters: int = -1, threshold: float = 0.0):
    # initializing max_distance
    max_distance = 0.0
    # defining the computation of the completeness condition

    def _condition():
        if n_clusters < 0:
            return max_distance > threshold
        else:
            return base_centroids.shape[0] < n_clusters
    # loop for completing the basis
    while True:
        # all distances from the basis of centroids
        distances = [distance_from_centroids(
            base_centroids, c) for c in other_centroids]
        # get the index of the maximum distance
        idx = np.argmax(distances)
        # get the maximum distance
        max_distance = distances[idx]
        if _condition():
            # add the new centroid --> (n_centroids, n_dimensions)
            base_centroids = np.concatenate(
                (base_centroids, [other_centroids[idx]]), axis=0)
        else:
            break
    return base_centroids


def check_weights_dict(weigths_dict):
    a = weigths_dict.copy()
    for k, v in a.items():
        if v.shape == torch.Size([0]):
            del weigths_dict[k]
    return weigths_dict


def generate_prob_labels(n_labels: int = 2, n_samples: int = 100, label: int = 0):
    if label < 0 or label > n_labels-1:
        raise ValueError(
            'The label value, label, cannot be lower than zero or higher than the number of labels minus one, n_labels-1')
    p = [float(0.2/(n_labels-1))]*n_labels
    p[label] = 0.8
    data = []
    for _ in range(n_samples):
        vec = np.bincount(np.random.choice(a=np.arange(
            n_labels), size=n_samples, p=p)) / n_samples
        if len(vec) == n_labels:
            data.append(vec)
    return data

def return_not_binary_indices(matrix):
    idx = []
    for i, column in enumerate(matrix.T):
        count = len(np.unique(column))
        if count > 2:
            idx.append(i)
    
    return idx

def get_dims_from_weights(weights):
    dims = [weights[0].shape[0]]
    for i in range(0, len(weights), 2):
        dims.append(weights[i].shape[1])
    return dims

def is_prime(n):
    return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))

def get_image_repr(z: int):
    current_is_prime = is_prime(z)
    add = 0
    # get how many to add for not to be prime
    while current_is_prime:
        add += 1
        current_is_prime = is_prime(z+add)
    dimensions = z+add
    for i in range(1, dimensions+1):
        left = dimensions % i
        if left == 0 and dimensions/i <= i:
            return i, int(dimensions/i), add

def get_square_image_repr(z: int):
    rad = np.round(np.sqrt(z))
    if z/rad > rad:
        rad +=1
    return int(rad), int(rad), int(rad**2-z)

def get_f_indices(
    n_samples: int,
    n_clients: int,
    client_id: int,
    balance: int = -1,
    seed: int = 51550,
    verbose: bool = False,
):
    if n_clients < 1:
        raise ValueError(
            'There must be more than one client')
    if client_id < 0 or client_id > n_clients-1:
        raise ValueError(
            'client_id must be in the range [0, (n_clients-1)], was given {} with n_clients {}'.
            format(client_id, n_clients))
    if balance < 0:
        r = uniform.rvs(balance, size=n_samples, random_state=seed)
        if verbose:
            print('Getting ids from uniform distribution')
    else:
        r = skewnorm.rvs(balance, size=n_samples, random_state=seed)
        if verbose:
            print('Getting ids from skewed gaussian distribution')
    hist = np.histogram(r, bins=n_clients)[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    start = np.sum(hist[:client_id])
    return idx[start:start+hist[client_id]]
