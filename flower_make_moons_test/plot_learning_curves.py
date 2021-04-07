#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:15:49 2021

@author: relogu
"""
#%% importations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import pathlib

#%% definitions of constant and functions
single_model = 'single model'
clients_mean = 'clients\' mean'

def process_df1(df):
    """Aggregate all clients to be store as a mean for seaborn."""
    for client in df['client'].unique():
        if client != single_model:
            s = clients_mean
            df.loc[df['client']==client, 'client']=s
    
    return df

def process_df(df):
    """Compute the mean from all cients and add it to the dataframe."""
    n_clients = len(df['client'].unique())-1
    for client in df['client'].unique():
        if client == 'l_curve_nofed' or client == 'nofed':
            s = single_model
        else:
            s = 'client '+str(client[-1:])
        df.loc[df['client']==client, 'client']=s
    
    mean = df[df['client']==single_model].reset_index().copy()
    mean['client']=clients_mean
    mean['loss']=0.0
    mean['accuracy']=0.0
    clients = df[df['client']!=single_model].reset_index().copy()
    for client in clients['client'].unique():
        mean['loss']=mean['loss']+clients[clients['client']==client].reset_index()['loss']/n_clients
        mean['accuracy']=mean['accuracy']+clients[clients['client']==client].reset_index()['accuracy']/n_clients
    
    client_list = df[df['client']!=single_model]['client'].unique()
    client = client_list[0]
    mean = df[df['client']==client].reset_index().copy()
    for client in client_list[1:]:
        mean = mean.append(df[df['client']==client].reset_index().copy(), ignore_index = True)
    
    mean['client'] = clients_mean
    df = df[df['client']==single_model].reset_index()
    df = df.append(clients, ignore_index = True)
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
    elif feature == 'plus_same':
        feature = 'advanced FL'
    elif feature[2:] == 'transf_same':
        feature = 'FL&TL'
    return n_clients, feature

#%% main
if __name__ == "__main__":

    path = '../RESULTS/'
    folders =  glob.glob(path+'*')
    print('Listed folders')
    mean = None
    # _same folders 
    for folder in folders:
        if folder[-3:] == 'png': continue
        if folder[11:13] != 'FL': continue
        #if folder[-2:] != 'tr': continue
        #flavor = folder[-2:]
        #if folder[-4:] != 'same': continue
        #flavor = folder[-4:]
        #if folder[-3:] != 'rot': continue
        #flavor = folder[-3:]
        if folder[-11:] != 'transf_same': continue
        flavor = folder[-11:]
        #if folder[-9:] != 'plus_same': continue
        #flavor = folder[-9:]
        files = glob.glob(folder+'/*.dat')
        #print('Listed files in '+str(folder))
        conv = pd.read_csv(files[0], index_col=False)
        for file in files[1:]:
            #print('Reading '+str(file))
            conv = conv.append(pd.read_csv(file, index_col=False), ignore_index = True)
        conv = conv.append(pd.read_csv(path+'single_model_'+flavor+'/l_curve_nofed.dat', index_col=False), ignore_index = True)
        n_clients, feature = read_simulation_from_folder(folder)
        
        print('Processing dataframe')
        conv = process_df(conv)
        print('Plotting learning curves')
        title = 'Simulation with '+str(n_clients)+' clients '+feature+' dataset'
        filename = folder+'/accuracy.png'
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel('accuracy')
        plt.xlabel("round")
        sns.lineplot(x='round', y='accuracy', hue='client', data=conv)#, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.show(block=False)
        plt.savefig(filename)
        plt.close()
    
        filename = folder+'/loss.png'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel('loss')
        plt.xlabel("round")
        sns.lineplot(x='round', y='loss', hue='client', data=conv, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.show(block=False)
        plt.savefig(filename)
        plt.close()
        
        print('Processing dataframe again')
        conv = conv[conv['client']!=clients_mean].reset_index().copy()
        conv = process_df1(conv)
        m = conv[conv['client']=='clients\' mean'].copy()
        m['client'] = str(n_clients)+' '+m['client']
        if mean is None: mean = m
        else: mean = mean.append(m)
        '''
        print('Plotting reduced learning curves')
        filename = folder+'/accuracy_red.png'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel('accuracy')
        plt.xlabel("round")
        sns.lineplot(x='round', y='accuracy', hue='client', data=conv, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.show(block=False)
        plt.savefig(filename)
        plt.close()
    
        filename = folder+'/loss_red.png'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel('loss')
        plt.xlabel("round")
        sns.lineplot(x='round', y='loss', hue='client', data=conv, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.show(block=False)
        plt.savefig(filename)
        plt.close()
        '''
    
    mean = mean.append(conv[conv['client']==single_model].copy(), ignore_index = True)
    mean = mean.sort_values('client')
    title = 'Comparison between set ups with different # clients'
    filename = path+'single_model_'+flavor+'/accuracy_red.png'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    plt.title(title)
    ax.set_ylabel('accuracy')
    plt.xlabel("round")
    sns.lineplot(x='round', y='accuracy', hue='client', data=mean, ci='sd')#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
    plt.draw()
    #plt.show(block=False)
    plt.savefig(filename)
    plt.close()

    filename = path+'single_model_'+flavor+'/loss_red.png'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    plt.title(title)
    ax.set_ylabel('loss')
    plt.xlabel("round")
    sns.lineplot(x='round', y='loss', hue='client', data=mean, ci='sd')#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
    plt.draw()
    #plt.show(block=False)
    plt.savefig(filename)
    plt.close()

# %%
