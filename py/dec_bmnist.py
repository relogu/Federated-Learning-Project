#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
from py.dumping.output import dump_result_dict
from py.dec.util import (create_autoencoder, create_clustering_model, target_distribution)
from losses import get_keras_loss_names, get_keras_loss
import py.metrics as my_metrics
from flwr.common.typing import Parameters
from sklearn.cluster import KMeans
from tensorflow.keras.initializers import RandomNormal, VarianceScaling, GlorotUniform
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow_addons.metrics as tfa_metrics
import argparse
import os
import pathlib
import numpy as np
import pickle

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_parser():
    parser = argparse.ArgumentParser(description="UDE Training Script")
    parser.add_argument("--batch_size", dest="batch_size",
                        default=256, type=int, help="Batch size")
    parser.add_argument("--hardware_acc", dest="cuda_flag", action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("--lim_cores", dest="lim_cores",
                        action='store_true', help="Flag for limiting cores")
    parser.add_argument("--folder", dest="out_folder",
                        type=type(str('')), help="Folder to output images")
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
                        default=140,
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
    tf.config.set_visible_devices(gpus[2:4], 'GPU')
    
    # preparing dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    n_features = int(x_train.shape[1]*x_train.shape[2])
    x_train, x_test = np.round(x_train.reshape(x_train.shape[0], n_features)/255), np.round(x_test.reshape(x_test.shape[0], n_features)/255)
    
    # initializing common configuration dict
    initial_learning_rate = 0.1
    
    def lr_step_decay(epoch, lr):
        # lr is divided by 10 every 20000 rounds
        drop_rate = 10
        epoch_drop = 20000
        lr = initial_learning_rate
        if epoch > epoch_drop:
            lr = initial_learning_rate/drop_rate
        if epoch > 2*epoch_drop:
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
            500,
            500,
            2000,
            args.n_clusters],  # DEC paper proportions
        # 'relu' --> DEC paper # 'selu' --> is better for binary
        'act': 'selu',
        'ae_metrics': [my_metrics.rounded_accuracy,
                       'accuracy',
                       tfa_metrics.HammingLoss(mode='multilabel', threshold=0.50)],
        'cl_optimizer': SGD(
            learning_rate=args.cl_lr,
            momentum=0.9),
        'cl_epochs': args.cl_epochs,
        'update_interval': args.update_interval,
        'ae_loss': get_keras_loss(args.ae_loss),
        'cl_loss': 'kld',
        'seed': args.seed}
    
    print('AE loss is {}'.format(config['ae_loss']))

    up_frequencies = np.array([np.array(np.count_nonzero(
        x_train[:, i])/x_train.shape[0]) for i in range(n_features)])

    # pre-train the autoencoder
    pretrained_weights = path_to_out/'encoder.npz'
    if not pretrained_weights.exists():
        print('There are no existing weights in the output folder for the autoencoder')
        
        autoencoder, encoder, decoder = create_autoencoder(config, up_frequencies)
        
        autoencoder.compile(
            metrics=config['ae_metrics'],
            optimizer=config['ae_optimizer'],
            loss=config['ae_loss']
        )
        
        # fitting the autoencoder
        history = autoencoder.fit(x=x_train,
                                  y=x_train,
                                  batch_size=config['batch_size'],
                                  epochs=int(config['ae_epochs']),
                                  validation_data=(x_test, x_test),
                                  #callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)],
                                  verbose=2)
        with open(path_to_out/'pretrain_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder', parameters)

    trained_weights = path_to_out/'encoder_ft.npz'
    if not trained_weights.exists():
        param: Parameters = np.load(pretrained_weights, allow_pickle=True)
        weights = param['arr_0']
        # no dropout, keep denoising
        config['dropout'] = 0.0
        config['ran_flip'] = 0.0
        autoencoder, encoder, decoder = create_autoencoder(config, None)

        encoder.set_weights(weights)
        
        autoencoder.compile(
            metrics=config['ae_metrics'],
            optimizer=config['ae_optimizer'],
            loss=config['ae_loss']
        )
        
        # fitting again the autoencoder
        history = autoencoder.fit(x=x_train,
                                  y=x_train,
                                  batch_size=config['batch_size'],
                                  epochs=int(2*config['ae_epochs']),
                                  validation_data=(x_test, x_test),
                                  #callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)],
                                  verbose=2)
        with open(path_to_out/'finetune_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder_ft', parameters)

    # clean from the auxialiary layer for the clustering model
    autoencoder, encoder, decoder = create_autoencoder(config, None)

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
    # saving the model weights
    parameters = np.array([kmeans.cluster_centers_])
    print('Saving initial centroids')
    np.savez(path_to_out/'initial_centroids', parameters)

    # training the clustering model
    clustering_model = create_clustering_model(
        config['n_clusters'],
        encoder)
    # compiling the clustering model
    clustering_model.compile(
        optimizer=config['cl_optimizer'],
        loss=config['cl_loss'])
    clustering_model.get_layer(
        name='clustering').set_weights(np.array([kmeans.cluster_centers_]))
    y_old = None
    print('Initializing the target distribution')
    q = clustering_model.predict(x_train, verbose=0)
    # update the auxiliary target distribution p
    p = target_distribution(q)
    train_loss, eval_loss = 0.1, 0
    # for i in range(int(config['cl_epochs'])):
    i = 0
    while True:
        i += 1
        if i % config['update_interval'] == 1:
            # if train_loss < eval_loss:
            print('Updating the target distribution')
            train_q = clustering_model.predict(x_train, verbose=0)
            # update the auxiliary target distribution p
            train_p = target_distribution(train_q)
        clustering_model.fit(x=x_train,
                             y=train_p,
                             verbose=2,
                             batch_size=config['batch_size'])
        # evaluation
        q = clustering_model.predict(x_train, verbose=0)
        # update the auxiliary target distribution p
        p = target_distribution(q)
        # retrieving loss
        loss = clustering_model.evaluate(x_train, p, verbose=2)
        # evaluate the clustering performance using some metrics
        y_pred = q.argmax(1)
        # getting the cycle accuracy of evaluation set
        x_ae_test = autoencoder(x_train)
        y_ae_pred = clustering_model.predict(
            np.round(x_ae_test), verbose=0).argmax(1)
        cycle_acc = my_metrics.acc(y_pred, y_ae_pred)
        del y_ae_pred, x_ae_test
        # evaluating metrics
        result = {}
        if y_train is not None and args.verbose:
            acc = my_metrics.acc(y_train, y_pred)
            nmi = my_metrics.nmi(y_train, y_pred)
            print('DEC Clustering\nEpoch %d\n\tacc %.5f\n\tnmi %.5f' %
                  (i, acc, nmi))
            print('Cycle accuracy is {}'.format(cycle_acc))
            # dumping and retrieving the results
            metrics = {'accuracy': acc,
                       'normalized_mutual_info_score': nmi,}
            result = metrics.copy()
        result['cycle_accuracy'] = cycle_acc
        result['loss'] = eval_loss
        result['round'] = i
        # check for required convergence
        if i > 1:
            tol = float(1 - my_metrics.acc(y_pred, y_old))
            if i % 100 and args.verbose:
                print("Current label change ratio is {}, i.e. {}/{} samples".
                      format(tol, int(tol*len(x_train)), len(x_train)))
            if tol < 0.001:  # and eval_cycle_acc > 0.9:# and i > 2000: # from DEC paper
                print("Final label change ratio is {}, i.e. {}/{} samples, reached after {} iteration".
                      format(tol, int(tol*len(x_train)), len(x_train), i))
                break
            else:
                y_old = y_pred.copy()
        else:
            tol = 1
            y_old = y_pred.copy()
        result['tol'] = tol
        dump_result_dict('clustering_model', result,
                         path_to_out=path_to_out)

    # saving the model weights
    parameters = np.array(clustering_model.get_weights(), dtype=object)
    np.savez(path_to_out/'clustering', parameters)
