#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:15:49 2021

@author: relogu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import pathlib

path = '/home/relogu/Desktop/OneDrive/UNIBO/Magistrale/Federated Learning Project/RESULTS/'
folders =  glob.glob(path+'*')
for folder in folders:
    files = glob.glob(folder+'/*.dat')
    conv = pd.read_csv(files[0], index_col=False)
    for file in files[1:]:
        conv = conv.append(pd.read_csv(file, index_col=False), ignore_index = True)
        
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    plt.title(file)
    ax.set_ylabel('accuracy')
    plt.xlabel("round")
    sns.lineplot(x='round', y='accuracy', hue='client', data=conv, style='client')#, markers=['.', '.'])
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    plt.title(file)
    ax.set_ylabel('loss')
    plt.xlabel("round")
    sns.lineplot(x='round', y='loss', hue='client', data=conv, style='client')#, markers=['.', '.'])
    plt.show()