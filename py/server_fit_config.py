#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:11:01 2021

@author: relogu
"""


def simple_clustering_on_fit_config(rnd: int,
                                    ae_epochs: int = 300,
                                    kmeans_epochs: int = 20,
                                    cl_epochs: int = 1000):
    if rnd < ae_epochs+1:
        return {'model': 'pretrain_ae',
                'first': (rnd == 1),
                'last': (rnd == ae_epochs),
                'actual_round': rnd,
                'total_rounds': ae_epochs}
    elif rnd < ae_epochs+kmeans_epochs+1:
        return {'model': 'k-means',
                'first': (rnd == ae_epochs+1),
                'last': (rnd == ae_epochs+1),
                'actual_round': rnd-ae_epochs,
                'total_rounds': kmeans_epochs}
    else:
        return {'model': 'clustering',
                'first': (rnd == ae_epochs+kmeans_epochs+1),
                'last': (rnd == cl_epochs+ae_epochs+1),
                'actual_round': rnd-ae_epochs-kmeans_epochs,
                'total_rounds': cl_epochs}


def udec_clustering_on_fit_config(rnd: int,
                                  ae_epochs: int = 300,
                                  cl_epochs: int = 1000,
                                  n_clusters: int = 2):
    if rnd == 1:
        config = {'model': 'freq_avg',
                  'n_clusters': n_clusters,
                  'first': (rnd == 1),
                  'last': (rnd == 1),
                  'actual_round': rnd,
                  'total_rounds': 1}
    elif rnd < ae_epochs+2:
        config = {'model': 'pretrain_ae',
                  'n_clusters': n_clusters,
                  'first': (rnd == 2),
                  'last': (rnd == ae_epochs),
                  'actual_round': rnd-1,
                  'total_rounds': ae_epochs}
    elif rnd < int(3*ae_epochs+2):
        config = {'model': 'finetune_ae',
                  'n_clusters': n_clusters,
                  'first': (rnd == ae_epochs+2),
                  'last': (rnd == int(3*ae_epochs+1)),
                  'actual_round': rnd-ae_epochs-1,
                  'total_rounds': int(2*ae_epochs)}
    elif rnd < int(3*ae_epochs+3):
        config = {'model': 'k-means',
                  'n_clusters': n_clusters,
                  'first': (rnd == int(3*ae_epochs+2)),
                  'last': (rnd == int(3*ae_epochs+2)),
                  'actual_round': int(rnd-1-3*ae_epochs),
                  'total_rounds': 1}
    else:
        config = {'model': 'clustering',
                  'n_clusters': n_clusters,
                  'first': (rnd == int(3*ae_epochs+3)),
                  'last': (rnd == int(cl_epochs+3*ae_epochs+2)),
                  'actual_round': int(rnd-2-3*ae_epochs),
                  'total_rounds': cl_epochs}
    return config


def kfed_clustering_on_fit_config(rnd: int,
                                  ae_epochs: int = 300,
                                  cl_epochs: int = 1000,
                                  n_clusters: int = 2):
    if rnd == 1:
        config = {'model': 'freq_avg',
                  'n_clusters': n_clusters,
                  'first': (rnd == 1),
                  'last': (rnd == 1),
                  'actual_round': rnd,
                  'total_rounds': 1}
    elif rnd < ae_epochs+2:
        config = {'model': 'pretrain_ae',
                  'n_clusters': n_clusters,
                  'first': (rnd == 2),
                  'last': (rnd == ae_epochs),
                  'actual_round': rnd-1,
                  'total_rounds': ae_epochs}
    elif rnd < ae_epochs+3:
        config = {'model': 'k-means',
                  'n_clusters': n_clusters,
                  'first': (rnd == ae_epochs+2),
                  'last': (rnd == ae_epochs+2),
                  'actual_round': rnd-1-ae_epochs,
                  'total_rounds': 1}
    else:
        config = {'model': 'clustering',
                  'n_clusters': n_clusters,
                  'first': (rnd == ae_epochs+3),
                  'last': (rnd == cl_epochs+ae_epochs+2),
                  'actual_round': rnd-ae_epochs-2,
                  'total_rounds': cl_epochs}
    return config


def clustergan_on_fit_config(rnd: int,
                             total_epochs: int):
    return {'model': 'clustergan',
            'actual_rounds': rnd,
            'total_epochs': total_epochs}


def simple_kmeans_on_fit_config(rnd: int,
                                kmeans_epochs: int = 20):
    if rnd < kmeans_epochs+1:
        return {'model': 'k-means'}
