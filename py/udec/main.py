#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
import argparse
import os
import pathlib
import numpy as np
import sys
import pickle

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss

from flwr.common.typing import Parameters

import py.dataset_util as data_util
import py.metrics as my_metrics
from py.udec.util import create_denoising_autoencoder, create_tied_denoising_autoencoder, create_prob_autoencoder, create_tied_prob_autoencoder, create_clustering_model, target_distribution
from py.dumping.plots import print_confusion_matrix
from py.dumping.output import dump_pred_dict, dump_result_dict

out_1 = 'UDEC\nEpoch %d/%d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f'


def get_parser():
    parser = argparse.ArgumentParser(description="UDE Training Script")
    parser.add_argument("--batch_size", dest="batch_size",
                        default=64, type=int, help="Batch size")
    parser.add_argument("--hardware_acc", dest="cuda_flag", action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("--lim_cores", dest="lim_cores",
                        action='store_true', help="Flag for limiting cores")
    parser.add_argument("--folder", dest="out_folder",
                        type=type(str('')), help="Folder to output images")
    parser.add_argument('--groups', dest='groups',
                        required=True,
                        action='append',
                        help='which groups of variables to use for EUROMDS dataset')
    parser.add_argument('--ex_col', dest='ex_col', required=True,
                        action='append', help='which columns to exclude for EUROMDS dataset')
    parser.add_argument("--n_clusters", dest="n_clusters", default=10,
                        type=int, help="Define the number of clusters to identify")
    parser.add_argument('--fold_n',
                        dest='fold_n',
                        required=False,
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        action='store',
                        help='fold number for train-test partitioning')
    parser.add_argument('--shuffle',
                        dest='shuffle',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='wheater to shuffle in train-test partitioning')
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=False,
                        type=int,
                        default=50000,
                        action='store',
                        help='number of epochs for the autoencoder pre-training')
    parser.add_argument('--cl_epochs',
                        dest='cl_epochs',
                        required=False,
                        type=int,
                        default=10000,
                        action='store',
                        help='number of epochs for the clustering step')
    parser.add_argument('--binary',
                        dest='binary',
                        required=False,
                        action='store_true',
                        help='Flag for using probabilistic binary neurons')
    parser.add_argument('--tied',
                        dest='tied',
                        required=False,
                        action='store_true',
                        help='Flag for using tied layers in autoencoder')
    parser.add_argument('--plotting',
                        dest='plotting',
                        required=False,
                        action='store_true',
                        help='Flag for plotting confusion matrix')
    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=0.05,
                        required=False,
                        action='store',
                        help='Flag for dropout layer in autoencoder')
    parser.add_argument('--ran_flip',
                        dest='ran_flip',
                        type=float,
                        default=0.25,
                        required=False,
                        action='store',
                        help='Flag for RandomFlipping layer in autoencoder')
    parser.add_argument('--ortho',
                        dest='ortho',
                        required=False,
                        action='store_true',
                        help='Flag for orthogonality regularizer in autoencoder (tied only)')
    parser.add_argument('--u_norm',
                        dest='u_norm',
                        required=False,
                        action='store_true',
                        help='Flag for unit norm constraint in autoencoder (tied only)')
    parser.add_argument('--cl_lr',
                        dest='cl_lr',
                        required=False,
                        type=float,
                        default=0.1,
                        action='store',
                        help='clustering model learning rate')
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=100,
                        action='store',
                        help='set the update interval for the clusters distribution')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Flag for verbosity')
    return parser
    

if __name__ == "__main__":
    # configuration
    # get parameters
    args = get_parser().parse_args()
    # disable possible gpu devices (add hard acc, selection)
    if not args.cuda_flag:
        print('No CUDA')
        tf.config.set_visible_devices([], 'GPU')
    elif args.lim_cores:
        print('Limiting CPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    # defining output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    os.makedirs(path_to_out, exist_ok=True)
    # initializing common configuration dict
    config = {
        'batch_size': args.batch_size,
        'splits': 5,
        'fold_n': args.fold_n,
        'n_clusters': args.n_clusters,
        'shuffle': args.shuffle,
        'kmeans_epochs': 300,
        'kmeans_n_init': 25,
        'ae_epochs': args.ae_epochs,
        'ae_lr': 0.1,#0.01, # DEC paper
        'ae_momentum': 0.9,
        'cl_lr': args.cl_lr,
        'cl_momentum': 0.9,
        'cl_epochs': args.cl_epochs,
        'update_interval': args.update_interval,
        # bce (specific for binary, overfit w/o dropout),#'mse' (no overfit w/o dropout)#,#(general)
        'ae_loss': 'binary_crossentropy',
        'cl_loss': 'kld',
        'seed': args.seed}

    # preparing dataset
    for g in args.groups:
        if g not in data_util.EUROMDS_GROUPS:
            print('One of the given groups is not allowed.\nAllowed groups: {}'.
                  format(data_util.EUROMDS_GROUPS))
            sys.exit()
    for c in args.ex_col:
        if c not in data_util.get_euromds_cols():
            print('One of the given columns is not allowed.\nAllowed columns: {}'.
                  format(data_util.get_euromds_cols()))
            sys.exit()
    # getting the entire dataset
    x = data_util.get_euromds_dataset(
        groups=args.groups, exclude_cols=args.ex_col)
    # getting the number of features
    n_features = len(x.columns)
    x = np.array(x)
    # getting labels from HDP
    prob = data_util.get_euromds_dataset(groups=['HDP'])
    y = []
    for label, row in prob.iterrows():
        if np.sum(row) > 0:
            y.append(row.argmax())
        else:
            y.append(-1)
    y = np.array(y)
    # getting the outcomes
    outcomes = data_util.get_outcome_euromds_dataset()
    outcomes = np.array(outcomes[['outcome_3', 'outcome_2']])
    # getting IDs
    ids = data_util.get_euromds_ids()
    # setting the autoencoder layers
    dims = [x.shape[-1],
            int((2/3)*(n_features)),
            int((2/3)*(n_features)),
            int((2.5)*(n_features)),
            args.n_clusters]  # DEC paper proportions
    # init = VarianceScaling(scale=1. / 3.,
    #                        mode='fan_in',
    #                        distribution="uniform") # old
    init = RandomNormal(mean=0.0,
                        stddev=0.2) #stddev=0.01) # DEC paper, is better

    config['ae_dims'] = dims
    config['ae_init'] = init
    config['ae_act'] = 'selu' # 'relu' --> DEC paper # 'selu' --> is better for binary

    # define the splitting
    train_idx, test_idx = data_util.split_dataset(
        x=x,
        splits=config['splits'],
        shuffle=config['shuffle'],
        fold_n=config['fold_n'],
        r_state=config['seed'])
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train, y_test = None, None
    outcomes_train, outcomes_test = None, None
    id_train, id_test = None, None
    if y is not None:
        y_test = y[test_idx]
        y_train = y[train_idx]
    if outcomes is not None:
        outcomes_train = outcomes[train_idx]
        outcomes_test = outcomes[test_idx]
    if ids is not None:
        id_train = ids[train_idx]
        id_test = ids[test_idx]
        
    up_frequencies = np.array([np.array(np.count_nonzero(
        x_train[:, i])/x_train.shape[0]) for i in range(n_features)])

    # pre-train the autoencoder
    pretrained_weights = path_to_out/'encoder.npz'
    if not pretrained_weights.exists():
        print('There are no existing weights in the output folder for the autoencoder')
        if args.binary:
            if args.tied:
                autoencoder, encoder, decoder = create_tied_prob_autoencoder(
                    config['ae_dims'], init=config['ae_init'], dropout_rate=args.dropout, act=config['ae_act'])
            else:
                autoencoder, encoder, decoder = create_prob_autoencoder(
                    config['ae_dims'], init=config['ae_init'], dropout_rate=args.dropout, act=config['ae_act'])
        else:
            print('Freq :{}'.format(up_frequencies))
            if args.tied:
                autoencoder, encoder, decoder = create_tied_denoising_autoencoder(
                    config['ae_dims'], up_freq=up_frequencies, init=config['ae_init'],
                    dropout_rate=args.dropout, act=config['ae_act'],
                    ortho=args.ortho, u_norm=args.u_norm, noise_rate=args.ran_flip)
            else:
                autoencoder, encoder, decoder = create_denoising_autoencoder(
                    config['ae_dims'], up_freq=up_frequencies, init=config['ae_init'],
                    dropout_rate=args.dropout, act=config['ae_act'])
        # ae_optimizer = SGD(learning_rate=config['ae_lr'],
        #                    momentum=config['ae_momentum'],
        #                    decay=(config['ae_lr']-0.0001)/config['ae_epochs'])  # old
        ae_optimizer = SGD(
            learning_rate=config['ae_lr'],
            momentum=config['ae_momentum'],
            decay=float(9/((3/5)*int(config['ae_epochs']))))  # from DEC paper
        autoencoder.compile(
            metrics=[my_metrics.rounded_accuracy, 'accuracy'],
            optimizer=ae_optimizer,
            loss=config['ae_loss']
        )
        # fitting the autoencoder
        history = autoencoder.fit(x=x_train,
                                  y=x_train,
                                  batch_size=config['batch_size'],
                                  validation_data=(x_test, x_test),
                                  epochs=int(config['ae_epochs']),
                                  verbose=1)
        with open(path_to_out/'pretrain_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder', parameters)
    
    trained_weights = path_to_out/'encoder_ft.npz'
    if not trained_weights.exists():
        param: Parameters = np.load(pretrained_weights, allow_pickle=True)
        weights = param['arr_0']
        # no dropout, keep denoising
        if args.binary:
            if args.tied:
                autoencoder, encoder, decoder = create_tied_prob_autoencoder(
                    config['ae_dims'], act=config['ae_act'])
            else:
                autoencoder, encoder, decoder = create_prob_autoencoder(
                    config['ae_dims'], act=config['ae_act'])
        else:
            if args.tied:
                autoencoder, encoder, decoder = create_tied_denoising_autoencoder(
                    config['ae_dims'], noise_rate=0.0, act=config['ae_act'],
                    ortho=args.ortho, u_norm=args.u_norm)
            else:
                autoencoder, encoder, decoder = create_denoising_autoencoder(
                    config['ae_dims'], noise_rate=0.0, act=config['ae_act'])
        
        encoder.set_weights(weights)
        
        # ae_optimizer = SGD(learning_rate=config['ae_lr'],
        #                    momentum=config['ae_momentum'],
        #                    decay=(config['ae_lr']-0.0001)/config['ae_epochs'])  # old
        ae_optimizer = SGD(
            learning_rate=config['ae_lr'],
            momentum=config['ae_momentum'],
            decay=float(9/((3/5)*int(config['ae_epochs']))))  # from DEC paper
        
        autoencoder.compile(
            metrics=[my_metrics.rounded_accuracy, 'accuracy'],
            optimizer=ae_optimizer,
            loss=config['ae_loss']
        )
        autoencoder.summary()
        # fitting again the autoencoder
        history = autoencoder.fit(x=x_train,
                                    y=x_train,
                                    batch_size=config['batch_size'],
                                    validation_data=(x_test, x_test),
                                    epochs=int(2*config['ae_epochs']),
                                    verbose=1)
        with open(path_to_out/'finetune_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder_ft', parameters)

    # clean from the auxialiary layer for the clustering model
    if args.binary:
        if args.tied:
            autoencoder, encoder, decoder = create_tied_prob_autoencoder(
                config['ae_dims'], act=config['ae_act'])
        else:
            autoencoder, encoder, decoder = create_prob_autoencoder(
                config['ae_dims'], act=config['ae_act'])
    else:
        if args.tied:
            autoencoder, encoder, decoder = create_tied_denoising_autoencoder(
                config['ae_dims'], act=config['ae_act'], noise_rate=0.0,
                ortho=args.ortho, u_norm=args.u_norm)
        else:
            autoencoder, encoder, decoder = create_denoising_autoencoder(
                config['ae_dims'], act=config['ae_act'])
                
    param: Parameters = np.load(trained_weights, allow_pickle=True)
    weights = param['arr_0']
    encoder.set_weights(weights)

    # get an estimate for clusters centers using k-means
    kmeans = KMeans(init='k-means++',
                    n_clusters=config['n_clusters'],
                    # number of different random initializations
                    n_init=config['kmeans_n_init'],
                    random_state=config['seed'])
    # fitting clusters' centers using k-means
    kmeans.fit(encoder.predict(x_train))

    # training the clustering model
    clustering_model = create_clustering_model(
        config['n_clusters'],
        encoder)
    # compiling the clustering model
    cl_optimizer = SGD(
        learning_rate=config['cl_lr'],
        momentum=config['cl_momentum'])
    clustering_model.compile(
        optimizer=cl_optimizer,
        loss=config['cl_loss'])
    clustering_model.get_layer(
        name='clustering').set_weights(np.array([kmeans.cluster_centers_]))
    y_old = None
    for i in range(int(config['cl_epochs'])):
        if i % config['update_interval'] == 0:
            print('Updating the target distribution')
            q = clustering_model.predict(x_train, verbose=0)
            # update the auxiliary target distribution p
            p = target_distribution(q)
        history = clustering_model.fit(x=x_train, y=p, verbose=2,
                                       batch_size=config['batch_size'])
        # evaluation
        q_eval = clustering_model.predict(x_test, verbose=0)
        # update the auxiliary target distribution p
        p_eval = target_distribution(q_eval)
        # retrieving loss
        loss = clustering_model.evaluate(x_test, p_eval, verbose=2)
        # evaluate the clustering performance using some metrics
        y_pred = q_eval.argmax(1)
        # evaluating metrics
        result = {}
        if y_test is not None and args.verbose:
            acc = my_metrics.acc(y_test, y_pred)
            nmi = my_metrics.nmi(y_test, y_pred)
            ami = my_metrics.ami(y_test, y_pred)
            ari = my_metrics.ari(y_test, y_pred)
            ran = my_metrics.ran(y_test, y_pred)
            homo = my_metrics.homo(y_test, y_pred)
            if args.plotting and i % 10 == 0:  # print confusion matrix
                print_confusion_matrix(
                    y_test, y_pred,
                    path_to_out=path_to_out)
            print(out_1 % (i+1, int(config['cl_epochs']), acc, nmi, ami, ari, ran, homo))
            # dumping and retrieving the results
            metrics = {"accuracy": acc,
                       "normalized_mutual_info_score": nmi,
                       "adjusted_mutual_info_score": ami,
                       "adjusted_rand_score": ari,
                       "rand_score": ran,
                       "homogeneity_score": homo}
            result = metrics.copy()
        result['eval_loss'] = loss
        result['train_loss'] = history.history['loss'][0]
        result['round'] = i+1
        dump_result_dict('clustering_model', result,
                        path_to_out=path_to_out)
        if id_test is not None:
            pred = {'ID': id_test,
                    'label': y_pred}
            dump_pred_dict('pred', pred,
                           path_to_out=path_to_out)
        # check for required convergence
        if i > 1:
            tol = 1 - my_metrics.acc(y_pred, y_old)
            if i%100:
                print("Current label change ratio is {}".format(tol))
            if tol < 0.001: # from DEC paper
                print("Final label change ratio is {}, i.e. {}/{} samples, reached at {} iteration". \
                    format(tol, int(tol*len(x_test)), len(x_test), i))
                break
            else:
                y_old = y_pred.copy()
        else:
            y_old = y_pred.copy()
            

    # saving the model weights
    parameters = np.array(clustering_model.get_weights(), dtype=object)
    np.savez(path_to_out/'clustering', parameters)
    
