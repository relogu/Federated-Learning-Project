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

def process_df1(df):
    """Aggregate all clients to be store as a mean for seaborn."""
    for client in df['client'].unique():
        if client != 'single model':
            s = 'clients\' mean'
            df.loc[df['client']==client, 'client']=s
    
    return df

def process_df(df):
    """Compute the mean from all cients and add it to the dataframe."""
    n_clients = len(df['client'].unique())-1
    for client in df['client'].unique():
        if client == 'nofed':
            s = 'single model'
        else:
            s = 'client '+str(client[-1:])
        df.loc[df['client']==client, 'client']=s
    
    mean = df[df['client']=='single model'].copy()
    mean['client']='clients\' mean'
    mean['loss']=0.0
    mean['accuracy']=0.0
    clients = df[df['client']!='single model']
    for client in clients['client'].unique():
        mean['loss']=mean['loss']+clients[clients['client']==client].reset_index()['loss']/n_clients
        mean['accuracy']=mean['accuracy']+clients[clients['client']==client].reset_index()['accuracy']/n_clients
    df = df.append(mean, ignore_index = True)
    
    return df
    

def read_simulation_from_folder(folderpath):
    """Read the folder to extract the simulation's features."""
    strs = str.split(folderpath, '/')
    n_clients = int(strs[-1:][0][3])
    feature = str(strs[-1:][0][12:])
    if feature == 'same':
        feature = 'standard'
    elif feature == 'rot':
        feature = 'rotated'
    elif feature == 'tr':
        feature = 'traslated'
    return n_clients, feature

if __name__ == "__main__":

    path = '/home/relogu/Desktop/OneDrive/UNIBO/Magistrale/Federated Learning Project/RESULTS/'
    folders =  glob.glob(path+'*')
    for folder in folders:
        files = glob.glob(folder+'/*.dat')
        conv = pd.read_csv(files[0], index_col=False)
        for file in files[1:]:
            conv = conv.append(pd.read_csv(file, index_col=False), ignore_index = True)
        
        n_clients, feature = read_simulation_from_folder(folder)
        
        conv = process_df(conv)
        
        title = 'Simulation with '+str(n_clients)+' clients '+feature+' dataset'
            
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        plt.title(title)
        ax.set_ylabel('accuracy')
        plt.xlabel("round")
        sns.lineplot(x='round', y='accuracy', hue='client', data=conv)#, style='client')#, markers=['.', '.'])
        plt.show()
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        plt.title(title)
        ax.set_ylabel('loss')
        plt.xlabel("round")
        sns.lineplot(x='round', y='loss', hue='client', data=conv, style='client')#, markers=['.', '.'])
        plt.show()
        
        conv = conv[conv['client']!='clients\' mean']
        conv = process_df1(conv)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        plt.title(title)
        ax.set_ylabel('accuracy')
        plt.xlabel("round")
        sns.lineplot(x='round', y='accuracy', hue='client', data=conv, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.show()
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        plt.title(title)
        ax.set_ylabel('loss')
        plt.xlabel("round")
        sns.lineplot(x='round', y='loss', hue='client', data=conv, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.show()
