#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
import os
import pathlib
import numpy as np
import pickle

from py.dumping.output import dump_result_dict
from py.dec.util import (create_clustering_model, create_denoising_autoencoder, target_distribution)
import py.metrics as my_metrics
from py.parsers import dec_mnist_parser
from py import compute_centroid_np

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.initializers import RandomNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from flwr.common.typing import Parameters
from sklearn.cluster import KMeans


if __name__ == "__main__":
    # get parameters
    args = dec_mnist_parser().parse_args()
    print('Arguments passed: {}'.format(args))
    np.random.seed(51550)
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

    # Restrict keras to use only selected GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print('Physical devices: {}'.format(gpus))
    print('GPU(s) chosen: {}'.format(args.gpus))
    gpus = [gpus[g] for g in args.gpus]
    tf.config.set_visible_devices(gpus, 'GPU')
    gpus = tf.config.list_logical_devices('GPU')
    print('Logical devices: {}'.format(gpus))
    # strategy = tf.distribute.experimental.CentralStorageStrategy(
    #     compute_devices=gpus[1:],
    #     parameter_device=gpus[0]
    # )
    # strategy = tf.distribute.MirroredStrategy(gpus)

    # preparing dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    n_features = int(x_train.shape[1]*x_train.shape[2])
    x_train, x_test = x_train.reshape(
        x_train.shape[0], n_features)/255, x_test.reshape(x_test.shape[0], n_features)/255

    # initializing common configuration dict
    initial_learning_rate = 0.1

    def lr_step_decay(epoch, lr):
        # lr is divided by 10 every 20000 rounds
        drop_rate = 10
        epoch_drop = 20000
        lr = initial_learning_rate
        if epoch > epoch_drop:
            lr = initial_learning_rate/(drop_rate**int(epoch/epoch_drop))
        return lr

    config = {
        'batch_size': 256,
        'n_clusters': 10,
        'kmeans_n_init': 20,
        'ae_epochs': 50000,
        'ae_optimizer': SGD(
            learning_rate=0.1,
            momentum=0.9,
            decay=(0.1-0.0001)/50000),
        'ae_optimizer': Adam(),
        # 'ae_dims': [
        #     784,  # input
        #     500,  # first layer
        #     500,  # second layer
        #     2000,  # third layer
        #     10,  # output (feature space)
        # ],
        'ae_act': 'relu',
        'ae_init': RandomNormal(mean=0.0,
                                stddev=0.01),
        # 'ae_init': GlorotUniform(seed=51550),
        'is_tied': args.tied,
        'u_norm_reg': args.u_norm,
        'ortho_w_con': args.ortho,
        'uncoll_feat_reg': args.uncoll,
        'use_bias': args.use_bias,
        'dropout_rate': args.dropout,
        'noise_rate': args.noise,
        'ran_flip_conf': None,
        'ae_metrics': [
            my_metrics.rounded_accuracy,
        ],
        'cl_optimizer': SGD(
            learning_rate=0.01,
            momentum=0.9),
        'update_interval': args.update_interval,
        'ae_loss': 'mse',
        'cl_loss': 'kld',
        'seed': args.seed}

    # TODO: Gready Layer-Wise pretrain of the autoencoder necessary
    pretrained_weights = path_to_out/'encoder.npz'
    if not pretrained_weights.exists():
        print('There are no existing weights in the output folder for the autoencoder')

        # with strategy.scope():
        autoencoder, encoder, decoder = create_denoising_autoencoder(
            flavor='real',
            dims=config['ae_dims'],
            activation=config['ae_act'],
            w_init=config['ae_init'],
            is_tied=config['is_tied'],
            u_norm_reg=config['u_norm_reg'],
            ortho_w_con=config['ortho_w_con'],
            uncoll_feat_reg=config['uncoll_feat_reg'],
            use_bias=config['use_bias'],
            dropout_rate=config['dropout_rate'],
            noise_rate=config['noise_rate'],
            ran_flip_conf=None,
            )

        print(autoencoder.summary())

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
                                  callbacks=[  # LearningRateScheduler(lr_step_decay, verbose=1),
                                      EarlyStopping(
                                          patience=5000,
                                          verbose=1,
                                          mode="auto",
                                          baseline=None,
                                          restore_best_weights=False,)
                                  ],
                                  verbose=2)
        with open(path_to_out/'pretrain_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder', parameters)

    trained_weights = path_to_out/'encoder_ft.npz'
    if not trained_weights.exists():
        param = np.load(pretrained_weights, allow_pickle=True)
        weights = np.array([param[p] for p in param])[0]
        print('There are no existing weights in the output folder for the autoencoder')

        # with strategy.scope():
        autoencoder, encoder, decoder = create_denoising_autoencoder(
            flavor='real',
            dims=config['ae_dims'],
            activation=config['ae_act'],
            is_tied=config['is_tied'],
            u_norm_reg=config['u_norm_reg'],
            ortho_w_con=config['ortho_w_con'],
            uncoll_feat_reg=config['uncoll_feat_reg'],
            use_bias=config['use_bias'],
            dropout_rate=0.0,
            noise_rate=0.0,
            ran_flip_conf=None,
            )

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
                                  callbacks=[  # LearningRateScheduler(lr_step_decay, verbose=1),
                                      EarlyStopping(
                                          patience=5000,
                                          verbose=1,
                                          mode="auto",
                                          baseline=None,
                                          restore_best_weights=False,)
                                  ],
                                  verbose=2)
        with open(path_to_out/'finetune_ae_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder_ft', parameters)

    # with strategy.scope():
    autoencoder, encoder, decoder = create_denoising_autoencoder(
        flavor='real',
        dims=config['ae_dims'],
        activation=config['ae_act'],
        is_tied=config['is_tied'],
        u_norm_reg=config['u_norm_reg'],
        ortho_w_con=config['ortho_w_con'],
        uncoll_feat_reg=config['uncoll_feat_reg'],
        use_bias=config['use_bias'],
        dropout_rate=0.0,
        noise_rate=0.0,
        ran_flip_conf=None,
        )

    param = np.load(trained_weights, allow_pickle=True)
    weights = np.array([param[p] for p in param])[0]
    encoder.set_weights(weights)

    # get an estimate for clusters centers using k-means
    z = encoder(x_train).numpy()
    kmeans = KMeans(
        init='k-means++',
        n_clusters=config['n_clusters'],
        # number of different random initializations
        n_init=config['kmeans_n_init'],
    ).fit(z)
    initial_labels = kmeans.labels_
    # saving the model weights
    centroids = []
    n_classes = len(np.unique(initial_labels))
    for i in np.unique(initial_labels):
        idx = (initial_labels == i)
        centroids.append(compute_centroid_np(z[idx, :]))
    # saving the model weights
    centroids = np.array(centroids)
    print('Saving initial centroids')
    np.savez(path_to_out/'initial_centroids', centroids)
    print('Shape of centroids layer {}'.format(np.array([centroids]).shape))

    # with strategy.scope():
    # training the clustering model
    clustering_model = create_clustering_model(
        n_clusters=config['n_clusters'],
        encoder=encoder,
        alpha=config['n_clusters']-1)
    # compiling the clustering model
    clustering_model.compile(
        optimizer=config['cl_optimizer'],
        loss=config['cl_loss'])
    clustering_model.get_layer(
        name='clustering').set_weights(np.array([kmeans.cluster_centers_]))
    y_old = initial_labels
    train_loss, eval_loss = 0.1, 0
    i = 0
    while True:
        i += 1
        # if i % config['update_interval'] == 1:
        #     # if train_loss < eval_loss:
        print('Shuffling data')
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx, :]
        y_old = y_old[idx]
        y_train = y_train[idx]
        print('Updating the target distribution')
        train_q = clustering_model(x_train).numpy()
        # update the auxiliary target distribution p
        train_p = target_distribution(train_q)
        clustering_model.fit(x=x_train,
                             y=train_p,
                             verbose=2,
                             batch_size=config['batch_size'],
                             steps_per_epoch=config['update_interval'])
        # evaluation
        q = clustering_model(x_train).numpy()
        y_pred = q.argmax(1)
        # update the auxiliary target distribution p
        p = target_distribution(q)
        # retrieving loss
        loss = clustering_model.evaluate(x_train, p, verbose=2)
        # test
        q = clustering_model(x_test).numpy()
        # update the auxiliary target distribution p
        p = target_distribution(q)
        # retrieving loss
        loss = clustering_model.evaluate(x_test, p, verbose=2)
        # getting the cycle accuracy of evaluation set
        x_ae_test = autoencoder(x_test)
        y_ae_pred = clustering_model(
            np.round(x_ae_test)).numpy().argmax(1)
        cycle_acc = my_metrics.acc(y_pred, y_ae_pred)
        del y_ae_pred, x_ae_test
        # evaluating metrics
        result = {}
        if y_test is not None and args.verbose:
            acc = my_metrics.acc(y_test, y_pred)
            nmi = my_metrics.nmi(y_test, y_pred)
            print('DEC Clustering\nEpoch %d\n\tacc %.5f\n\tnmi %.5f' %
                  (i, acc, nmi))
            print('Cycle accuracy is {}'.format(cycle_acc))
            # dumping and retrieving the results
            metrics = {'accuracy': acc,
                       'normalized_mutual_info_score': nmi, }
            result = metrics.copy()
        result['cycle_accuracy'] = cycle_acc
        result['loss'] = eval_loss
        result['round'] = i
        # check for required convergence
        #tol = float(1 - my_metrics.acc(y_old, y_pred))
        print(y_old == y_pred)
        tol = float(1 - np.sum(y_old == y_pred)/len(x_train))
        if args.verbose:
            print("Current label change ratio is {}, i.e. {}/{} samples".
                  format(tol, int(tol*len(x_train)), len(x_train)))
        if tol < 0.001:  # and eval_cycle_acc > 0.9:# and i > 2000: # from DEC paper
            print("Final label change ratio is {}, i.e. {}/{} samples, reached after {} iteration".
                  format(tol, int(tol*len(x_train)), len(x_train), i))
            break
        else:
            y_old = y_pred.copy()
        result['tol'] = tol
        dump_result_dict('clustering_model', result,
                         path_to_out=path_to_out)

        # saving the model weights
        parameters = np.array(encoder.get_weights(), dtype=object)
        np.savez(path_to_out/'encoder_final', parameters)

        parameters = np.array(clustering_model.get_layer(
            name='clustering').get_weights(), dtype=object)
        np.savez(path_to_out/'final_centroids', parameters)

        break
    
    print('Configuration dict: {}'.format(config))
