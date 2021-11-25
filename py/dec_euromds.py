#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""

from dumping.output import dump_pred_dict, dump_result_dict
from dumping.plots import print_confusion_matrix
from dec.util import (create_autoencoder, create_clustering_model, target_distribution)
from util import compute_centroid_np, return_not_binary_indices
from losses import get_keras_loss_names, get_keras_loss
import metrics as my_metrics
import dataset_util as data_util
from flwr.common.typing import Parameters
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from tensorflow.keras.initializers import RandomNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow_addons.metrics as tfa_metrics
import argparse
import os
import pathlib
import numpy as np
import sys
import pickle

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    parser.add_argument('--fill',
                        dest='fill',
                        required=False,
                        action='store_true',
                        help='Flag for fill NaNs in dataset')
    parser.add_argument("--n_clusters", dest="n_clusters", default=10,
                        type=int, help="Define the number of clusters to identify")
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=False,
                        type=int,
                        default=50000,
                        action='store',
                        help='number of epochs for the autoencoder pre-training')
    parser.add_argument('--ae_loss',
                        dest='ae_loss',
                        required=True,
                        type=type(''),
                        default='mse',
                        choices=get_keras_loss_names(),
                        action='store',
                        help='Loss function for autoencoder training')
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
                        default=0.20,
                        required=False,
                        action='store',
                        help='Flag for dropout layer in autoencoder')
    parser.add_argument('--ran_flip',
                        dest='ran_flip',
                        type=float,
                        default=0.20,
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
                        default=0.01,
                        action='store',
                        help='clustering model learning rate')
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=20,
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
    
    # Restrict keras to use only 2 GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print('Physical devices: {}'.format(gpus))
    tf.config.set_visible_devices(gpus[:4], 'GPU')
    gpus = tf.config.list_logical_devices('GPU')
    print('Logical devices: {}'.format(gpus))
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():

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
        print("Filling? {}".format(args.fill))
        if args.fill:
            fill = 2044
        else:
            fill = 0
        x = data_util.get_euromds_dataset(
            groups=args.groups, exclude_cols=args.ex_col, accept_nan=fill, fill_fn=data_util.fillcolumn_prob, verbose=args.verbose)
        # getting the number of features
        n_features = len(x.columns)
        print('Number of features extracted is {}'.format(n_features))
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
        
        # initializing common configuration dict
        initial_learning_rate = 0.1
        
        def lr_step_decay(epoch, lr):
            # lr is divided by 10 every 20000 rounds
            drop_rate = 10
            epoch_drop = int((2/5)*args.ae_epochs)
            lr = initial_learning_rate
            if epoch >= epoch_drop:
                lr = initial_learning_rate/drop_rate
            if epoch >= 2*epoch_drop:
                lr = lr/drop_rate
            return lr
        
        config = {
            'batch_size': args.batch_size,
            'n_clusters': args.n_clusters,
            'kmeans_epochs': 300,
            'kmeans_n_init': 25,
            'binary': args.binary,
            'tied': args.tied,
            'dropout': args.dropout,
            'ortho': args.ortho,
            'u_norm': args.u_norm,
            'ran_flip': args.ran_flip,
            'use_bias': True,
            'ae_epochs': args.ae_epochs,
            # 'ae_optimizer': SGD(learning_rate=config['ae_lr'],
            #                    momentum=config['ae_momentum'],
            #                    decay=(config['ae_lr']-0.0001)/config['ae_epochs']) , # old
            'ae_optimizer': SGD(
                learning_rate=0.1,
                momentum=0.9,#),
                decay=(0.1-0.0001)/args.ae_epochs),
            # 'init': VarianceScaling(scale=1. / 2.,#3.,
            #                        mode='fan_in',
            #                        distribution="uniform"), # old
            # 'init': RandomNormal(mean=0.0,
            #                     stddev=0.2)  # stddev=0.01), # DEC paper, is better
            'init': GlorotUniform(seed=51550),
            'dims': [n_features,
                    150,#500,#int((2)*(n_features)),#int((2/3)*(n_features)),
                    150,#500,#int((2)*(n_features)),#int((2/3)*(n_features)),
                    500,#2000,#int((3)*(n_features)),#int((2.5)*(n_features)),
                    5],#args.n_clusters],  # DEC paper proportions
            # 'relu' --> DEC paper # 'selu' --> is better for binary
            'act': 'selu',
            # 'ae_metrics': [my_metrics.rounded_accuracy,
            #                'accuracy',
            #                tfa_metrics.HammingLoss(mode='multilabel', threshold=0.50)],
            'cl_optimizer': SGD(
                learning_rate=args.cl_lr,
                momentum=0.9),
            'cl_epochs': args.cl_epochs,
            'update_interval': args.update_interval,
            'ae_loss': get_keras_loss(args.ae_loss),
            'cl_loss': 'kld',
            'seed': args.seed}
        
        print('AE loss is {}'.format(config['ae_loss']))
        
        nb_idx = return_not_binary_indices(x)
        b_idx = list(range(len(x[0,:])))[len(nb_idx):]
        config['b_idx'] = b_idx
        config['ae_metrics'] = [
            my_metrics.get_rounded_accuracy(idx=b_idx),
            my_metrics.get_slice_accuracy(idx=nb_idx),
            my_metrics.get_slice_hamming_loss(mode='multilabel', threshold=0.50, idx=b_idx),
            my_metrics.get_slice_log_mse_loss(idx=nb_idx),
            'mse']

        up_frequencies = np.array([np.array(np.count_nonzero(
            x[:, i])/x.shape[0]) for i in b_idx])

        # pre-train the autoencoder
        pretrained_weights = path_to_out/'encoder.npz'
        if not pretrained_weights.exists():
            print('There are no existing weights in the output folder for the autoencoder')
            
            autoencoder, encoder, decoder = create_autoencoder(config, up_frequencies)
            autoencoder.summary()
            
            autoencoder.compile(
                metrics=config['ae_metrics'],
                optimizer=config['ae_optimizer'],
                loss=config['ae_loss']
            )
            
            # fitting the autoencoder
            history = autoencoder.fit(x=x,
                                    y=x,
                                    batch_size=config['batch_size'],
                                    epochs=int(config['ae_epochs']),
                                    callbacks=[
                                        #   LearningRateScheduler(
                                        #       lr_step_decay, verbose=1),
                                        #   EarlyStopping(
                                        #       patience=1000,
                                        #       verbose=1,
                                        #       mode="auto",
                                        #       baseline=None,
                                        #       restore_best_weights=False,)
                                    ],
                                    verbose=2)
            with open(path_to_out/'pretrain_ae_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            parameters = np.array(encoder.get_weights(), dtype=object)
            np.savez(path_to_out/'encoder', parameters)

        # no dropout, keep denoising
        config['dropout'] = 0.0
        config['ran_flip'] = 0.0

        trained_weights = path_to_out/'encoder_ft.npz'
        if not trained_weights.exists():
            param: Parameters = np.load(pretrained_weights, allow_pickle=True)
            weights = param['arr_0']
            autoencoder, encoder, decoder = create_autoencoder(config, None)
            autoencoder.summary()
        
            encoder.set_weights(weights)
            
            autoencoder.compile(
                metrics=config['ae_metrics'],
                optimizer=config['ae_optimizer'],
                loss=config['ae_loss']
            )

            autoencoder.summary()
            # fitting again the autoencoder
            history = autoencoder.fit(x=x,
                                    y=x,
                                    batch_size=config['batch_size'],
                                    epochs=int(2*config['ae_epochs']),
                                    callbacks=[
                                        #   LearningRateScheduler(
                                        #       lr_step_decay, verbose=1),
                                        #   EarlyStopping(
                                        #       patience=1000,
                                        #       verbose=1,
                                        #       mode="auto",
                                        #       baseline=None,
                                        #       restore_best_weights=False,)
                                    ],
                                    verbose=2)
            with open(path_to_out/'finetune_ae_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            parameters = np.array(encoder.get_weights(), dtype=object)
            np.savez(path_to_out/'encoder_ft', parameters)

        # clean from the auxialiary layer for the clustering model
        autoencoder, encoder, decoder = create_autoencoder(config, None)
        autoencoder.summary()
        encoder.summary()

        param: Parameters = np.load(trained_weights, allow_pickle=True)
        weights = np.squeeze(np.array([param[p] for p in param]))
        encoder.set_weights(weights)
        for w1, w2 in zip(encoder.get_weights(), autoencoder.get_weights()[:8]):
            print(np.sum(np.array(w1)- np.array(w2)))
        
        # get an estimate for clusters centers using k-means
        z = encoder(x).numpy()
        # # KMEANS
        # kmeans = KMeans(
        #     init='k-means++',
        #     n_clusters=config['n_clusters'],
        #     # number of different random initializations
        #     n_init=config['kmeans_n_init']
        # ).fit(z)
        # initial_labels = kmeans.labels_
        # TSNE+DBSCAN
        tsne = TSNE(n_components=2, random_state=51550).fit_transform(z)
        dbcl_tsne = DBSCAN(
            min_samples=40,
            eps=3,
            ).fit(tsne)
        initial_labels = dbcl_tsne.labels_
        centroids = []
        n_classes = len(np.unique(initial_labels))
        for i in np.unique(initial_labels):
            idx = (initial_labels == i)
            centroids.append(compute_centroid_np(z[idx,:]))
        # saving the model weights
        centroids = np.array(centroids)
        print('Saving initial centroids')
        np.savez(path_to_out/'initial_centroids', centroids)
        print('Shape of centroids layer {}'.format(np.array([centroids]).shape))

        # training the clustering model
        clustering_model = create_clustering_model(
            n_classes,#config['n_clusters'],
            encoder,
            alpha=int(config['n_clusters']-1))
        # compiling the clustering model
        clustering_model.compile(
            optimizer=config['cl_optimizer'],
            loss=config['cl_loss'])
        clustering_model.get_layer(
            name='clustering').set_weights(np.array([centroids]))
        clustering_model.summary()
        for w1, w2 in zip(encoder.get_weights(), clustering_model.get_weights()[:8]):
            print(np.sum(np.array(w1)- np.array(w2)))
            
        y_old = initial_labels
        train_loss, eval_loss = 0.1, 0
        # for i in range(int(config['cl_epochs'])):
        i = 0
        while True:
            i += 1
            # if i % config['update_interval'] == 1:
            #     # if train_loss < eval_loss:
            print('Shuffling data')
            idx = np.random.permutation(len(x))
            x = x[idx, :]
            y_old = y_old[idx]
            ids = ids[idx]
            print('Computing the target distribution')
            train_q = clustering_model(x).numpy()
            # update the auxiliary target distribution p
            train_p = target_distribution(train_q)
            clustering_model.fit(x=x,
                                y=train_p,
                                verbose=2,
                                #steps_per_epoch=config['update_interval'],
                                batch_size=config['batch_size'])
            # evaluation
            q = clustering_model(x).numpy()
            # update the auxiliary target distribution p
            p = target_distribution(q)
            # retrieving loss
            loss = clustering_model.evaluate(x, p, verbose=2)
            # evaluate the clustering performance using some metrics
            y_pred = q.argmax(1)
            # getting the cycle accuracy of evaluation set
            x_ae_test = autoencoder(x).numpy()
            y_ae_pred = clustering_model(x_ae_test).numpy().argmax(1)
            cycle_acc = my_metrics.acc(y_pred, y_ae_pred)
            del y_ae_pred, x_ae_test
            # evaluating metrics
            result = {}
            if y is not None and args.verbose:
                acc = my_metrics.acc(y, y_pred)
                nmi = my_metrics.nmi(y, y_pred)
                ami = my_metrics.ami(y, y_pred)
                ari = my_metrics.ari(y, y_pred)
                ran = my_metrics.ran(y, y_pred)
                homo = my_metrics.homo(y, y_pred)
                if args.plotting and i % 10 == 0:  # print confusion matrix
                    print_confusion_matrix(
                        y,
                        y_pred,
                        path_to_out=path_to_out)
                print('DEC Clustering\nEpoch %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' %
                    (i, acc, nmi, ami, ari, ran, homo))
                print('Cycle accuracy is {}'.format(cycle_acc))
                # dumping and retrieving the results
                metrics = {'accuracy': acc,
                        'normalized_mutual_info_score': nmi,
                        'adjusted_mutual_info_score': ami,
                        'adjusted_rand_score': ari,
                        'rand_score': ran,
                        'homogeneity_score': homo}
                result = metrics.copy()
            result['cycle_accuracy'] = cycle_acc
            result['loss'] = eval_loss
            result['round'] = i
            if ids is not None:
                pred = {'ID': ids,
                        'label': y_pred}
                dump_pred_dict('pred', pred,
                            path_to_out=path_to_out)
            # check for required convergence
            tol = float(1 - my_metrics.acc(y_old, y_pred))
            if i % 100 and args.verbose:
                print("Current label change ratio is {}, i.e. {}/{} samples".
                        format(tol, int(tol*len(x)), len(x)))
            if tol < 0.001:  # and eval_cycle_acc > 0.9:# and i > 2000: # from DEC paper
                print("Final label change ratio is {}, i.e. {}/{} samples, reached after {} iteration".
                        format(tol, int(tol*len(x)), len(x), i))
                break
            y_old = y_pred.copy()
            result['tol'] = tol
            dump_result_dict('clustering_model', result,
                            path_to_out=path_to_out)

        # saving the model weights
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder_final', parameters)

        parameters = np.array(clustering_model.get_layer(name='clustering').get_weights(), dtype=object)
        np.savez(path_to_out/'final_centroids', parameters)
