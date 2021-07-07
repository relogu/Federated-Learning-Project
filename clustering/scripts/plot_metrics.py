
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:41:54 2021

@author: relogu
"""
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from argparse import RawTextHelpFormatter

def parse_args():
    """Parse the arguments passed."""
    description = 'Script utility to plot curves'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--prefix',
                        dest='prefix',
                        required=True,
                        type=type(''),
                        action='store',
                        help='prefix to give to output images')
    _args = parser.parse_args()
    return _args

if __name__ == "__main__":

    # parsing arguments
    args = parse_args()
    prefix = args.prefix
    
    # %% plot metrics
    
    # reading output files
    path_to_file = pathlib.Path(__file__).parent.parent.absolute()
    output_folder = path_to_file/'output'
    images_folder = path_to_file/'results'/prefix
    images_folder.mkdir(exist_ok=True)
    # getting autoencoder results if exist
    clients = sorted(output_folder.glob('*_ae.dat'))
    ae_df = None
    if len(clients) > 0:
        # building dataframe of results
        ae_df = pd.DataFrame()
        for client in clients:
            c = pd.read_csv(client)
            ae_df = ae_df.append(c)
            client.unlink()
        ae_overall = ae_df.copy()
        ae_overall['client'] = 'all'
        ae_metrics = list(ae_df.columns)
        ae_metrics.remove('client')
        ae_metrics.remove('round')
    # getting iteration results
    clients = sorted(output_folder.glob('*.dat'))
    if len(clients) > 0:
        # building dataframe of results
        df = pd.DataFrame()
        for client in clients:
            c = pd.read_csv(client)
            df = df.append(c)
            client.unlink()
        overall = df.copy()
        overall['client'] = 'all'
        metrics = list(df.columns)
        metrics.remove('client')
        metrics.remove('round')
    '''
    # %%
    sns.set_style('whitegrid')
    for metric in metrics:
        title = metric.replace('_', ' ')
        filename = images_folder/metric
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel(metric)
        plt.xlabel("round")
        sns.lineplot(x='round', y=metric, hue='client', data=df)#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.savefig(filename)
        plt.close()
    '''
    # %%
    tmp = df.copy()
    tmp = tmp.append(overall)
    for metric in metrics:
        title = metric.replace('_', ' ')
        filename = images_folder/metric
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel(metric)
        plt.xlabel("round")
        sns.lineplot(x='round', y=metric, hue='client', data=tmp)#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.draw()
        plt.savefig(filename)
        plt.close()
    
    if ae_df is not None:
        tmp = ae_df.copy()
        tmp = tmp.append(ae_overall)
        for metric in ae_metrics:
            title = 'autoencoder '+metric
            filename = images_folder/str('autoencoder_'+metric)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
            plt.title(title)
            ax.set_ylabel(metric)
            plt.xlabel("round")
            sns.lineplot(x='round', y=metric, hue='client', data=tmp)#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
            plt.draw()
            plt.savefig(filename)
            plt.close()
    '''
    # %%
    for metric in metrics:
        title = metric.replace('_', ' ')
        filename = images_folder/metric
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        plt.title(title)
        ax.set_ylabel(metric)
        plt.xlabel("round")
        sns.lineplot(x='round', y=metric, hue='client', data=overall)#, palette=['Blue', 'Red'])#, style='client')#, markers=['.', '.'])
        plt.draw()
        #plt.savefig(filename)
        plt.close()
    '''
    # moving other images
    imgs = sorted(output_folder.glob('*'))
    for img in imgs:
        filename = img.name
        img.rename(images_folder/filename)
