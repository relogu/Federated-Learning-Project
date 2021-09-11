#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
import warnings
import math
import os
from itertools import chain as ichain
from typing import OrderedDict, Dict
import pathlib

import flwr as fl
import numpy as np
import torch
import torchvision
from py.clustergan.dense_model import DiscriminatorDense, EncoderDense, GeneratorDense
from py.clustergan.cnn_model import DiscriminatorCNN, EncoderCNN, GeneratorCNN
from py.clustergan.util import calc_gradient_penalty, sample_z

from flwr.client import NumPyClient
from flwr.common import (FitRes, Parameters, Scalar, Weights,
                         parameters_to_weights)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans
from sklearn.ensemble._hist_gradient_boosting import loss
from tensorflow.keras.optimizers import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import py.metrics as my_metrics
from py.dataset_util import split_dataset, PrepareData, PrepareDataSimple
from py.util import check_weights_dict
from py.udec.util import create_denoising_autoencoder, create_prob_autoencoder, create_clustering_model, target_distribution
from py.dumping.output import dump_pred_dict, dump_result_dict
from py.dumping.plots import print_confusion_matrix, plot_lifelines_pred

k_means_initializer = 'k-means++'
k_means_eval_string = 'Client %d, updated real accuracy of k-Means: %.5f'
out_1 = 'Client %d, FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f'
out_2 = 'Client %d, FedIter %d\n\tae_loss %.5f'
out_3 = 'Client %d, FedIter %d\n\tae_loss %.5f\n\taccuracy %.5f'
out_4 = 'Client %d, FedIter %d\n\tae_loss %.5f\n\taccuracy %.5f\n\tr_accuracy %.5f'
clustering_eval_string = 'Client %d, Acc = %.5f, nmi = %.5f, ari = %.5f ; loss = %.5f'

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class KFEDClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""
    
    def __init__(self,
                 x,  # data
                 y,  # labels
                 client_id,  # id of the client
                 config,  # configuration dict
                 seed: int = 51550):  # see for random gen
        # set
        self.n_clusters = config['n_clusters']
        self.ae_dims = config['ae_dims']
        self.ae_local_epochs = config['ae_local_epochs']
        self.ae_optimizer = SGD(
            lr=config['ae_lr'], momentum=config['ae_momentum'])
        self.ae_loss = config['ae_loss']
        self.cl_local_epochs = config['cl_local_epochs']
        self.cl_optimizer = SGD(
            lr=config['cl_lr'], momentum=config['cl_momentum'])
        self.cl_loss = config['cl_loss']
        self.update_interval = config['update_interval']
        self.kmeans_local_epochs = config['kmeans_local_epochs']
        self.kmeans_n_init = config['kmeans_n_init']
        if y is None:
            self.x_train, self.x_test = split_dataset(x)
            self.y_train = self.y_test = None
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = split_dataset(
                x, y)
        self.batch_size = config['batch_size']
        self.client_id = client_id
        self.seed = seed
        # default
        self.autoencoder = None
        self.encoder = None
        self.clustering_model = None
        self.f_round = 0
        self.p = None
        self.local_iter = 0
        self.step = None
        self.cluster_centers = None

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:
            return []
        elif self.step == 'pretrain_ae':
            return self.autoencoder.get_weights()
        elif self.step == 'k-FED':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _fit_clustering_model(self):
        for _ in range(int(self.cl_local_epochs)):
            if self.local_iter % self.update_interval == 0:
                q = self.clustering_model.predict(self.x_train, verbose=0)
                # update the auxiliary target distribution p
                self.p = target_distribution(q)
            self.clustering_model.fit(x=self.x_train, y=self.p, verbose=0)
            self.local_iter += 1

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self.step = config['model']
        print("Federated Round number %d, step: %s, rounds %d/%d" % (self.f_round,
                                                                     self.step,
                                                                     config['actual_round'],
                                                                     config['total_rounds']))
        if self.step == 'pretrain_ae':  # ae pretrain step
            if config['first']:
                # building and compiling autoencoder
                self.autoencoder, self.encoder = create_autoencoder(
                    self.ae_dims)
                self.autoencoder.compile(
                    optimizer=self.ae_optimizer,
                    loss=self.ae_loss
                )
            else:  # getting new weights
                self.autoencoder.set_weights(parameters)
            # fitting the autoencoder
            self.autoencoder.fit(x=self.x_train,
                                 y=self.x_train,
                                 batch_size=32,
                                 epochs=self.ae_local_epochs,
                                 verbose=0)
            # returning the parameters necessary for FedAvg
            return self.autoencoder.get_weights(), len(self.x_train), {}
        elif self.step == 'k-FED':  # k-Means step
            parameters = k_means_initializer  # k++ is used
            # number of cluster following the definition of heterogeneity
            n_cl = int(math.sqrt(self.n_clusters))
            self.kmeans = KMeans(init=parameters,
                                 n_clusters=n_cl,
                                 max_iter=self.kmeans_local_epochs,
                                 n_init=self.kmeans_n_init,  # number of different random initializations
                                 random_state=self.seed)
            # fitting clusters' centers using k-means
            self.kmeans.fit(self.encoder.predict(self.x_train))
            # returning the parameters necessary for k-FED
            return self.kmeans.cluster_centers_, len(self.x_train), {}
        elif self.step == 'clustering':
            if config['first']:  # initialize clustering layer with final kmeans' cluster centers
                # getting final clusters centers
                self.cluster_centers = parameters
                # initializing clustering model
                self.clustering_model = create_clustering_model(
                    self.n_clusters,
                    self.encoder)
                # compiling the clustering model
                self.clustering_model.compile(
                    optimizer=self.cl_optimizer,
                    loss=self.cl_loss)
                self.clustering_model.get_layer(
                    name='clustering').set_weights(np.array([self.cluster_centers]))
            else:  # getting new weights
                self.clustering_model.set_weights(parameters)
            # fitting clustering model
            self._fit_clustering_model()
            # returning the parameters necessary for FedAvg
            return self.clustering_model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = ()
        metrics = {}
        if self.step == 'pretrain_ae':
            # evaluation
            loss = self.autoencoder.evaluate(
                self.x_test, self.x_test, verbose=0)
            metrics = {"loss": loss}
            result = metrics.copy()
            result['client'] = self.client_id
            result['round'] = self.f_round
            dump_result_dict('client_'+str(self.client_id)+'_ae', result)
            print(out_2 % (self.client_id, self.f_round, loss))
            result = (loss, len(self.x_test), {})
        elif self.step == 'k-FED':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(
                self.encoder.predict(self.x_test))
            # computing metrics
            acc = my_metrics.acc(self.y_test, y_pred_kmeans)
            nmi = my_metrics.nmi(self.y_test, y_pred_kmeans)
            ami = my_metrics.ami(self.y_test, y_pred_kmeans)
            ari = my_metrics.ari(self.y_test, y_pred_kmeans)
            ran = my_metrics.ran(self.y_test, y_pred_kmeans)
            homo = my_metrics.homo(self.y_test, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                print_confusion_matrix(
                    self.y_test, y_pred_kmeans, client_id=self.client_id)
            # retrieving the results
            result = (loss, len(self.x_test), metrics)
        elif self.step == 'clustering':
            # evaluation
            q = self.clustering_model.predict(self.x_test, verbose=0)
            # update the auxiliary target distribution p
            p = target_distribution(q)
            # retrieving loss
            loss = self.clustering_model.evaluate(self.x_test, p, verbose=0)
            # evaluate the clustering performance using some metrics
            y_pred = q.argmax(1)
            if self.y_test is not None:
                acc = my_metrics.acc(self.y_test, y_pred)
                nmi = my_metrics.nmi(self.y_test, y_pred)
                ami = my_metrics.ami(self.y_test, y_pred)
                ari = my_metrics.ari(self.y_test, y_pred)
                ran = my_metrics.ran(self.y_test, y_pred)
                homo = my_metrics.homo(self.y_test, y_pred)
                if self.f_round % 10 == 0:  # print confusion matrix
                    print_confusion_matrix(
                        self.y_test, y_pred, client_id=self.client_id)
                print(out_1 % (self.client_id, self.f_round,
                               acc, nmi, ami, ari, ran, homo))
                # dumping and retrieving the results
                metrics = {"accuracy": acc,
                           "normalized_mutual_info_score": nmi,
                           "adjusted_mutual_info_score": ami,
                           "adjusted_rand_score": ari,
                           "rand_score": ran,
                           "homogeneity_score": homo}
                result = metrics.copy()
                result['loss'] = loss
                result['client'] = self.client_id
                result['round'] = self.local_iter
                dump_result_dict('client_'+str(self.client_id), result)
            result = (loss, len(self.x_test), metrics)
        return result


class SimpleKMeansClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 x,
                 y,
                 client_id,
                 config,
                 outcomes=None,
                 ids=None,
                 output_folder=None,
                 seed: int = 51550):
        # set
        train_idx, test_idx = split_dataset(
            x=x,
            splits=config['splits'],
            shuffle=config['shuffle'],
            fold_n=config['fold_n'])

        self.x_train, self.x_test = x[train_idx], x[test_idx]
        self.y_train = self.y_test = None
        self.outcomes_train = self.outcomes_test = None
        self.ids_train = self.ids_test = None
        if y is not None:
            self.y_train, self.y_test = y[train_idx], y[test_idx]
        if outcomes is not None:
            self.outcomes_train, self.outcomes_test = outcomes[train_idx], outcomes[test_idx]
        if ids is not None:
            self.ids_train, self.ids_test = ids[train_idx], ids[test_idx]
        self.client_id = client_id
        self.seed = seed
        self.kmeans_local_epochs = config['kmeans_local_epochs']
        self.n_clusters = config['n_clusters']
        self.kmeans_n_init = config['kmeans_n_init']
        # default
        self.kmeans = None
        self.f_round = 0
        self.p = None
        self.step = None
        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = pathlib.Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:  # first federated iteration
            return []
        elif self.step == 'k-means':
            return self.kmeans.cluster_centers_

    def _get_step(self, config):
        self.step = config['model']

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self._get_step(config)
        print("Federated Round number %d, step: %s" %
              (self.f_round, self.step))
        if self.step == 'k-means':  # k-Means step
            if parameters == []:
                parameters = k_means_initializer
                n_init = self.kmeans_n_init
            else:
                parameters = np.array(parameters)
                n_init = 1
            self.kmeans = KMeans(init=parameters,
                                 n_clusters=self.n_clusters,
                                 max_iter=self.kmeans_local_epochs,
                                 n_init=n_init,
                                 random_state=self.seed)
            # fitting clusters' centers using k-means
            self.kmeans.fit(self.x_train)
            # returning the parameters necessary for FedAvg
            return self.kmeans.cluster_centers_, len(self.x_train), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = (0.0, 1, {})
        if self.step == 'k-means' and self.y_test is not None:
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(self.x_test)
            # computing metrics
            acc = my_metrics.acc(self.y_test, y_pred_kmeans)
            nmi = my_metrics.nmi(self.y_test, y_pred_kmeans)
            ami = my_metrics.ami(self.y_test, y_pred_kmeans)
            ari = my_metrics.ari(self.y_test, y_pred_kmeans)
            ran = my_metrics.ran(self.y_test, y_pred_kmeans)
            homo = my_metrics.homo(self.y_test, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.f_round % 10 == 0:  # print confusion matrix
                print_confusion_matrix(
                    self.y_test, y_pred_kmeans, client_id=self.client_id,
                    path_to_out=self.out_dir)
            # plotting outcomes on the labels
            if self.outcomes_test is not None:
                times = self.outcomes_test[:, 0]
                events = self.outcomes_test[:, 1]
                plot_lifelines_pred(
                    times, events, y_pred_kmeans, client_id=self.client_id,
                    path_to_out=self.out_dir)
            if self.ids_test is not None:
                pred = {'ID': self.ids_test,
                        'label': y_pred_kmeans}
                dump_pred_dict('pred_client_'+str(self.client_id), pred,
                               path_to_out=self.out_dir)
            # dumping and retrieving the results
            metrics = {"accuracy": acc,
                       "normalized_mutual_info_score": nmi,
                       "adjusted_mutual_info_score": ami,
                       "adjusted_rand_score": ari,
                       "rand_score": ran,
                       "homogeneity_score": homo}
            result = metrics.copy()
            result['client'] = self.client_id
            result['round'] = self.f_round
            dump_result_dict('client_'+str(self.client_id), result,
                             path_to_out=self.out_dir)
            result = (loss, len(self.x_test), metrics)
        return result


class ClusterGANClient(NumPyClient):

    def __init__(self,
                 x,
                 y,
                 config,
                 ids=None,
                 outcomes=None,
                 client_id: int = 0,
                 hardw_acc_flag: bool = False,
                 output_folder=None
                 ):
        # Training details
        self.n_epochs = config['n_local_epochs']
        self.lr = config['learning_rate']
        self.b1 = config['beta_1']
        self.b2 = config['beta_2']
        self.decay = config['decay']
        self.n_skip_iter = config['d_step']

        # Data dimensions
        self.x_shape = config['x_shape']
        # Latent space info
        self.latent_dim = config['latent_dim']
        self.n_c = config['n_clusters']
        self.betan = config['betan']
        self.betac = config['betac']

        # Wasserstein+GP metric flag
        self.wass_metric = config['wass_metric']
        print('Using metric {}'.format(
            'Wassestrain' if self.wass_metric else 'Vanilla'))

        self.cuda = True if torch.cuda.is_available() and hardw_acc_flag else False
        self.device = torch.device(
            'cuda:0' if self.cuda else 'cpu')
        print('Using device {}'.format(self.device))
        torch.autograd.set_detect_anomaly(True)

        # Loss function
        self.bce_loss = torch.nn.BCELoss()
        self.xe_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        # Initialize NNs
        if config['conv_net']:
            self.generator = GeneratorCNN(self.latent_dim,
                                          self.n_c,
                                          self.x_shape)
            self.encoder = EncoderCNN(self.latent_dim,
                                      self.n_c)
            self.discriminator = DiscriminatorCNN(
                wass_metric=self.wass_metric)
        else:
            self.generator = GeneratorDense(latent_dim=self.latent_dim,
                                            n_c=self.n_c,
                                            gen_dims=config['gen_dims'],
                                            x_shape=self.x_shape,
                                            use_binary=config['use_binary'])
            self.encoder = EncoderDense(latent_dim=self.latent_dim,
                                        enc_dims=config['enc_dims'],
                                        n_c=self.n_c)
            self.discriminator = DiscriminatorDense(
                disc_dims=config['disc_dims'], wass_metric=self.wass_metric)

        if self.cuda:
            self.generator.cuda()
            self.encoder.cuda()
            self.discriminator.cuda()
            self.bce_loss.cuda()
            self.xe_loss.cuda()
            self.mse_loss.cuda()
        self.TENSOR = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Configure data loader
        self.batch_size = config['batch_size']
        train_idx, test_idx = split_dataset(
            x=x,
            splits=config['splits'],
            shuffle=config['shuffle'],
            fold_n=config['fold_n'])

        self.x_train = x[train_idx]
        self.y_train = y[train_idx]
        self.x_test = x[test_idx]
        self.y_test = y[test_idx]

        if outcomes is None or ids is None:
            self.trainloader = DataLoader(
                PrepareDataSimple(self.x_train, y=self.y_train),
                batch_size=self.batch_size)
            self.testloader = DataLoader(
                PrepareDataSimple(self.x_test, y=self.y_test),
                batch_size=self.batch_size)
        else:
            self.id_train = ids[train_idx]
            self.outcomes_train = outcomes[train_idx]
            self.id_test = ids[test_idx]
            self.outcomes_test = outcomes[test_idx]
            self.trainloader = DataLoader(
                PrepareData(x=self.x_train,
                            y=self.y_train,
                            ids=self.id_train,
                            outcomes=self.outcomes_train),
                batch_size=self.batch_size)
            self.testloader = DataLoader(
                PrepareData(x=self.x_test,
                            y=self.y_test,
                            ids=self.id_test,
                            outcomes=self.outcomes_test),
                batch_size=self.batch_size)

        self.ge_chain = ichain(self.generator.parameters(),
                               self.encoder.parameters())

        self.optimizer_GE = torch.optim.Adam(self.ge_chain,
                                             lr=self.lr,
                                             betas=(self.b1, self.b2),
                                             weight_decay=self.decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr,
                                            betas=(self.b1, self.b2))

        # ----------
        #  Training
        # ----------
        self.ge_l = []
        self.d_l = []
        self.c_zn = []
        self.c_zc = []
        self.c_i = []

        # metrics
        self.img_mse_loss = None
        self.lat_mse_loss = None
        self.lat_xe_loss = None

        # leghts of NN parameters to send and receive
        self.g_w_l = len(self.generator.state_dict().items())
        self.d_w_l = len(self.discriminator.state_dict().items())
        self.e_w_l = len(self.encoder.state_dict().items())

        # initiliazing to zero the federated epochs counter
        self.f_epoch = 0

        # for saving images
        self.save_images = config['save_images']
        self.client_id = client_id
        if output_folder is None:
            self.out_dir = output_folder
            self.img_dir = 'client_%d_images' % (self.client_id)
            os.makedirs(self.img_dir, exist_ok=True)
        else:
            self.out_dir = pathlib.Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)

    def train(self, config):
        # Training loop
        print('Begin training session with %i epochs...\n' % (self.n_epochs))
        for epoch in range(self.n_epochs):
            for i, (imgs, self.itruth_label, _, _) in enumerate(self.trainloader):

                # Ensure generator/encoder are trainable
                self.generator.train()
                self.encoder.train()

                # Zero gradients for models, resetting at each iteration because they sum up,
                # and we don't want them to pile up between different iterations
                self.generator.zero_grad()
                self.encoder.zero_grad()
                self.discriminator.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_GE.zero_grad()

                # Configure input
                self.real_imgs = Variable(imgs.type(self.TENSOR))

                # ---------------------------
                #  Train Generator + Encoder
                # ---------------------------

                # Sample random latent variables
                zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                          latent_dim=self.latent_dim,
                                          n_c=self.n_c,
                                          cuda=self.cuda)

                # Generate a batch of images
                gen_imgs = self.generator(zn, zc)

                # Discriminator output from real and generated samples
                gen_d = self.discriminator(gen_imgs)
                real_d = self.discriminator(self.real_imgs)
                valid = Variable(self.TENSOR(gen_imgs.size(
                    0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.TENSOR(gen_imgs.size(
                    0), 1).fill_(0.0), requires_grad=False)

                # Step for Generator & Encoder, n_skip_iter times less than for discriminator
                if (i % self.n_skip_iter == 0):
                    # Encode the generated images
                    enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = self.encoder(
                        gen_imgs)

                    # Calculate losses for z_n, z_c
                    zn_loss = self.mse_loss(enc_gen_zn, zn)
                    zc_loss = self.xe_loss(enc_gen_zc_logits, zc_idx)

                    # Check requested metric
                    if self.wass_metric:
                        # Wasserstein GAN loss
                        # ge_loss = torch.mean(gen_d) + betan * zn_loss + betac * zc_loss # original
                        # corrected
                        ge_loss = - \
                            torch.mean(gen_d) + self.betan * \
                            zn_loss + self.betac * zc_loss
                    else:
                        # Vanilla GAN loss
                        v_loss = self.bce_loss(gen_d, valid)
                        ge_loss = v_loss + self.betan * zn_loss + self.betac * zc_loss
                    # backpropagate the gradients
                    ge_loss.backward(retain_graph=True)
                    # computes the new weights
                    self.optimizer_GE.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Measure discriminator's ability to classify real from generated samples
                if self.wass_metric:
                    # Gradient penaltytorch.autograd.set_detect_anomaly(True) term
                    grad_penalty = calc_gradient_penalty(
                        self.discriminator, self.real_imgs, gen_imgs, cuda=self.cuda)

                    # Wasserstein GAN loss w/gradient penalty
                    # d_losss = torch.mean(real_d) - torch.mean(gen_d) + grad_penalty # original
                    # corrected
                    d_loss = - torch.mean(real_d) + \
                        torch.mean(gen_d) + grad_penalty

                else:
                    # Vanilla GAN loss
                    real_loss = self.bce_loss(real_d, valid)
                    fake_loss = self.bce_loss(gen_d, fake)
                    d_loss = (real_loss + fake_loss) / 2

                d_loss.backward(inputs=list(self.discriminator.parameters()))
                self.optimizer_D.step()
            # Save training losses
            self.d_l.append(d_loss.item())
            self.ge_l.append(ge_loss.item())
            print("[Federated Epoch %d/%d] [Client ID %d] [Epoch %d/%d] \n"
                  "\tModel Losses: [D: %f] [GE: %f]\n" %
                  (self.f_epoch,
                   config['total_epochs'],
                   self.client_id,
                   epoch+1,
                   self.n_epochs,
                   d_loss.item(),
                   ge_loss.item())
                  )

    def test(self, config):
        print('Begin evaluation session...\n')
        # Generator in eval mode
        self.generator.eval()
        self.encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        test_imgs, test_labels, test_ids, test_outcomes = next(
            iter(self.testloader))
        times = test_outcomes[:, 0]
        events = test_outcomes[:, 1]
        test_imgs = Variable(test_imgs.type(self.TENSOR))

        # Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = self.encoder(t_imgs)

        computed_labels = []
        for pred in e_tzc.detach().cpu().numpy():
            computed_labels.append(pred.argmax())
        computed_labels = np.array(computed_labels)

        # computing metrics
        acc = my_metrics.acc(t_label.detach().cpu().numpy(),
                             computed_labels)
        nmi = my_metrics.nmi(t_label.detach().cpu().numpy(),
                             computed_labels)
        ami = my_metrics.ami(t_label.detach().cpu().numpy(),
                             computed_labels)
        ari = my_metrics.ari(t_label.detach().cpu().numpy(),
                             computed_labels)
        ran = my_metrics.ran(t_label.detach().cpu().numpy(),
                             computed_labels)
        homo = my_metrics.homo(t_label.detach().cpu().numpy(),
                               computed_labels)
        print(out_1 % (self.client_id, self.f_epoch,
                       acc, nmi, ami, ari, ran, homo))
        # plotting outcomes on the labels
        # if self.outcomes_loader is not None:
        plot_lifelines_pred(
            times, events, computed_labels, client_id=self.client_id,
            path_to_out=self.out_dir)
        if self.f_epoch % 10 == 0:  # print confusion matrix
            print_confusion_matrix(
                t_label.detach().cpu().numpy(),
                computed_labels,
                client_id=self.client_id,
                path_to_out=self.out_dir)
        # dumping and retrieving the results
        metrics = {"accuracy": acc,
                   "normalized_mutual_info_score": nmi,
                   "adjusted_mutual_info_score": ami,
                   "adjusted_rand_score": ari,
                   "rand_score": ran,
                   "homogeneity_score": homo}
        result = metrics.copy()

        # Generate sample instances from encoding
        teg_imgs = self.generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        self.img_mse_loss = self.mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        self.c_i.append(self.img_mse_loss.item())

        # Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=self.latent_dim,
                                                 n_c=self.n_c,
                                                 cuda=self.cuda)
        # Generate sample instances
        gen_imgs_samp = self.generator(zn_samp, zc_samp)

        # Encode sample instances
        zn_e, zc_e, zc_e_logits = self.encoder(gen_imgs_samp)

        # Calculate cycle latent losses
        self.lat_mse_loss = self.mse_loss(zn_e, zn_samp)
        self.lat_xe_loss = self.xe_loss(zc_e_logits, zc_samp_idx)

        # Save latent space cycle losses
        self.c_zn.append(self.lat_mse_loss.item())
        self.c_zc.append(self.lat_xe_loss.item())

        # Save cycled and generated examples!
        if self.save_images:
            r_imgs, i_label = self.real_imgs.data[:
                                                  n_samp], self.itruth_label[:n_samp]
            e_zn, e_zc, e_zc_logits = self.encoder(r_imgs)
            reg_imgs = self.generator(e_zn, e_zc)
            save_image(reg_imgs.data[:n_samp],
                       self.img_dir/'cycle_reg_%06i.png' % (self.f_epoch),
                       nrow=n_sqrt_samp, normalize=True)
            save_image(gen_imgs_samp.data[:n_samp],
                       self.img_dir/'gen_%06i.png' % (self.f_epoch),
                       nrow=n_sqrt_samp, normalize=True)
            # Generate samples for specified classes
            stack_imgs = []
            for idx in range(self.n_c):
                # Sample specific class
                zn_samp, zc_samp, zc_samp_idx = sample_z(shape=self.n_c,
                                                         latent_dim=self.latent_dim,
                                                         n_c=self.n_c,
                                                         fix_class=idx,
                                                         cuda=self.cuda)
                # Generate sample instances
                gen_imgs_samp = self.generator(zn_samp, zc_samp)

                if (len(stack_imgs) == 0):
                    stack_imgs = gen_imgs_samp
                else:
                    stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)
            # Save class-specified generated examples!
            save_image(stack_imgs,
                       self.img_dir/'gen_classes_%06i.png' % (self.f_epoch),
                       nrow=self.n_c, normalize=True)

        print("[Federated Epoch %d/%d] [Client ID %d] \n"
              "\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]\n" %
              (self.f_epoch,
               config['total_epochs'],
               self.client_id,
               self.img_mse_loss.item(),
               self.lat_mse_loss.item(),
               self.lat_xe_loss.item())
              )

        result['img_mse_loss'] = self.img_mse_loss.item()
        result['lat_mse_loss'] = self.lat_mse_loss.item()
        result['lat_xe_loss'] = self.lat_xe_loss.item()
        result['client'] = self.client_id
        result['round'] = self.f_epoch
        dump_result_dict('client_'+str(self.client_id), result,
                         path_to_out=self.out_dir)
        pred = {'ID': test_ids,
                'label': computed_labels}
        dump_pred_dict('pred_client_'+str(self.client_id), pred,
                       path_to_out=self.out_dir)

    def get_parameters(self):
        g_par = np.array([val.cpu().numpy()
                 for _, val in self.generator.state_dict().items()], dtype=object)
        d_par = np.array([val.cpu().numpy()
                 for _, val in self.discriminator.state_dict().items()], dtype=object)
        e_par = np.array([val.cpu().numpy()
                 for _, val in self.encoder.state_dict().items()], dtype=object)
        parameters = np.concatenate([g_par, d_par, e_par], axis=0)
        return parameters

    def set_parameters(self, parameters):
        # generator
        g_par = parameters[:self.g_w_l].copy()
        params_dict = zip(self.generator.state_dict().keys(), g_par)
        g_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # discriminator
        d_par = parameters[self.g_w_l:int(self.g_w_l+self.d_w_l)].copy()
        params_dict = zip(self.discriminator.state_dict().keys(), d_par)
        d_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # encoder
        e_par = parameters[int(self.g_w_l+self.d_w_l):].copy()
        params_dict = zip(self.encoder.state_dict().keys(), e_par)
        e_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # checking for null weights
        g_state_dict = check_weights_dict(g_state_dict)
        d_state_dict = check_weights_dict(d_state_dict)
        e_state_dict = check_weights_dict(e_state_dict)
        # assigning weights
        self.generator.load_state_dict(g_state_dict, strict=True)
        self.discriminator.load_state_dict(d_state_dict, strict=True)
        self.encoder.load_state_dict(e_state_dict, strict=True)

    def fit(self, parameters, config):
        self.f_epoch += 1
        self.set_parameters(parameters)
        self.train(config)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.test(config)
        metrics = {"x": self.img_mse_loss.item(),
                   "z_n": self.lat_mse_loss.item(),
                   "z_c": self.lat_xe_loss.item()}
        return float(self.img_mse_loss.item()), len(self.testloader), metrics


class KMeansEmbedClusteringClient(NumPyClient):
    """Client object, to set client performed operations."""

    def __init__(self,
                 x,  # data
                 y,  # labels
                 client_id,  # id of the client
                 config,  # configuration dictionary
                 outcomes=None,  # outcomes for lifelines
                 ids=None,  # ids of data
                 output_folder=None,
                 seed: int = 51550):  # see for random gen
        # set
        self.n_clusters = config['n_clusters']
        self.ae_dims = config['ae_dims']
        self.ae_init = config['ae_init']
        self.ae_local_epochs = config['ae_local_epochs']
        self.ae_optimizer = SGD(
            lr=config['ae_lr'], momentum=config['ae_momentum'])
        self.ae_loss = config['ae_loss']
        self.cl_optimizer = SGD(
            learning_rate=config['cl_lr'], momentum=config['cl_momentum'])
        self.cl_local_epochs = config['cl_local_epochs']
        self.cl_optimizer = SGD(
            lr=config['cl_lr'], momentum=config['cl_momentum'])
        self.cl_loss = config['cl_loss']
        self.update_interval = config['update_interval']
        self.kmeans_n_init = config['kmeans_n_init']
        self.kmeans_local_epochs = config['kmeans_local_epochs']

        train_idx, test_idx = split_dataset(
            x=x,
            splits=config['splits'],
            shuffle=config['shuffle'],
            fold_n=config['fold_n'])

        self.x_train = x[train_idx]
        self.x_test = x[test_idx]
        self.y_train = self.y_test = None
        self.outcomes_train = self.outcomes_test = None
        self.id_train = self.id_test = None
        if y is not None:
            self.y_test = y[test_idx]
            self.y_train = y[train_idx]
        if outcomes is not None:
            self.outcomes_train = outcomes[train_idx]
            self.outcomes_test = outcomes[test_idx]
        if ids is not None:
            self.id_train = ids[train_idx]
            self.id_test = ids[test_idx]

        self.batch_size = config['batch_size']
        self.client_id = client_id
        self.seed = seed

        if output_folder is None:
            self.out_dir = output_folder
        else:
            self.out_dir = pathlib.Path(output_folder)
            os.makedirs(self.out_dir, exist_ok=True)

        # default
        self.autoencoder = None
        self.encoder = None
        self.clustering_model = None
        self.binary = config['binary']
        self.f_round = 0
        self.p = None
        self.local_iter = 0
        self.step = None
        self.cluster_centers = None
        self.plotting = config['plotting']
        self.last_histo = None

    def get_parameters(self):  # type: ignore
        """Get the model weights by model object."""
        if self.step is None:
            return []
        elif self.step == 'pretrain_ae':
            return self.autoencoder.get_weights()
        elif self.step == 'k-FED':
            return self.kmeans.cluster_centers_
        elif self.step == 'clustering':
            return self.clustering_model.get_weights()

    def _fit_clustering_model(self):
        for _ in range(int(self.cl_local_epochs)):
            if self.local_iter % self.update_interval == 0:
                q = self.clustering_model.predict(self.x_train, verbose=0)
                # update the auxiliary target distribution p
                self.p = target_distribution(q)
            self.clustering_model.fit(x=self.x_train, y=self.p, verbose=0)
        self.local_iter += 1

    def fit(self, parameters, config):  # type: ignore
        """Perform the fit step after having assigned new weights."""
        # increasing the number of epoch
        self.f_round += 1
        self.step = config['model']
        print("Federated Round number %d, step: %s, rounds %d/%d" % (self.f_round,
                                                                     self.step,
                                                                     config['actual_round'],
                                                                     config['total_rounds']))
        if self.step == 'pretrain_ae':  # ae pretrain step
            if config['first']:
                # building and compiling autoencoder
                if self.binary:
                    self.autoencoder, self.encoder, self.decoder = create_prob_autoencoder(
                        dims=self.ae_dims, init=self.ae_init)
                else:
                    up_frequencies = np.array([np.array(np.count_nonzero(self.x_train[:,i])/self.x_train.shape[0]) for i in range(self.x_train.shape[1])])
                    self.autoencoder, self.encoder, self.decoder = create_denoising_autoencoder(
                        dims=self.ae_dims, up_freq=up_frequencies, init=self.ae_init, act='selu')
                self.autoencoder.compile(
                    metrics=[my_metrics.rounded_accuracy, "accuracy"],
                    optimizer=self.ae_optimizer,
                    loss=self.ae_loss
                )
            else:  # getting new weights
                self.encoder.set_weights(parameters)
            # fitting the autoencoder
            self.last_histo = self.autoencoder.fit(x=self.x_train,
                                                   y=self.x_train,
                                                   batch_size=self.batch_size,
                                                   epochs=self.ae_local_epochs,
                                                   verbose=0)
            # returning the parameters necessary for FedAvg
            return self.encoder.get_weights(), len(self.x_train), {}
        elif self.step == 'k-means':  # k-Means step
            parameters = k_means_initializer  # k++ is used
            self.kmeans = KMeans(init=parameters,
                                 n_clusters=self.n_clusters,
                                 max_iter=self.kmeans_local_epochs,
                                 n_init=self.kmeans_n_init,  # number of different random initializations
                                 random_state=self.seed)
            # fitting clusters' centers using k-means
            self.kmeans.fit(self.encoder.predict(self.x_train))
            # returning the parameters necessary for k-FED
            return self.kmeans.cluster_centers_, len(self.x_train), {}
        elif self.step == 'clustering':
            if config['first']:  # initialize clustering layer with final kmeans' cluster centers
                # getting final clusters centers
                self.cluster_centers = parameters
                # initializing clustering model
                self.clustering_model = create_clustering_model(
                    self.n_clusters,
                    self.encoder)
                # compiling the clustering model
                self.clustering_model.compile(
                    optimizer=self.cl_optimizer,
                    loss=self.cl_loss)
                self.clustering_model.get_layer(
                    name='clustering').set_weights(np.array([self.cluster_centers]))
            else:  # getting new weights
                self.clustering_model.set_weights(parameters)
            # fitting clustering model
            self._fit_clustering_model()
            # returning the parameters necessary for FedAvg
            return self.clustering_model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        loss = 0.0
        acc = 0.0
        result = ()
        metrics = {}
        if self.step == 'pretrain_ae':
            # evaluation
            loss, r_accuracy, accuracy = self.autoencoder.evaluate(
                self.x_test, self.x_test, verbose=0)
            metrics = {"eval_loss": loss,
                       "eval_accuracy": accuracy,
                       "eval_r_accuracy": r_accuracy,
                       "train_loss": self.last_histo['loss'][-1],
                       "train_accuracy": self.last_histo['accuracy'][-1],
                       "train_r_accuracy": self.last_histo['rounded_accuracy'][-1]}
            result = metrics.copy()
            result['client'] = self.client_id
            result['round'] = self.f_round
            dump_result_dict('client_'+str(self.client_id)+'_ae', result,
                             path_to_out=self.out_dir)
            print(out_4 % (self.client_id, self.f_round, loss, accuracy, r_accuracy))
            result = (loss, len(self.x_test), {"r_accuracy": r_accuracy,"accuracy": accuracy})
        elif self.step == 'k-means':
            # predicting labels
            y_pred_kmeans = self.kmeans.predict(
                self.encoder.predict(self.x_test))
            # computing metrics
            acc = my_metrics.acc(self.y_test, y_pred_kmeans)
            nmi = my_metrics.nmi(self.y_test, y_pred_kmeans)
            ami = my_metrics.ami(self.y_test, y_pred_kmeans)
            ari = my_metrics.ari(self.y_test, y_pred_kmeans)
            ran = my_metrics.ran(self.y_test, y_pred_kmeans)
            homo = my_metrics.homo(self.y_test, y_pred_kmeans)
            print(out_1 % (self.client_id, self.f_round,
                  acc, nmi, ami, ari, ran, homo))
            if self.plotting and self.f_round % 10 == 0:
                print_confusion_matrix(
                    self.y_test, y_pred_kmeans, client_id=self.client_id,
                    path_to_out=self.out_dir)
            # retrieving the results
            result = (loss, len(self.x_test), metrics)
        elif self.step == 'clustering':
            # evaluation
            q = self.clustering_model.predict(self.x_test, verbose=0)
            # update the auxiliary target distribution p
            p = target_distribution(q)
            # retrieving loss
            loss = self.clustering_model.evaluate(self.x_test, p, verbose=0)
            # evaluate the clustering performance using some metrics
            y_pred = q.argmax(1)
            # plotting outcomes on the labels
            if self.plotting and self.outcomes_test is not None:
                times = self.outcomes_test[:, 0]
                events = self.outcomes_test[:, 1]
                plot_lifelines_pred(
                    times, events, y_pred, client_id=self.client_id,
                    path_to_out=self.out_dir)
            # evaluating metrics
            if self.y_test is not None:
                acc = my_metrics.acc(self.y_test, y_pred)
                nmi = my_metrics.nmi(self.y_test, y_pred)
                ami = my_metrics.ami(self.y_test, y_pred)
                ari = my_metrics.ari(self.y_test, y_pred)
                ran = my_metrics.ran(self.y_test, y_pred)
                homo = my_metrics.homo(self.y_test, y_pred)
                if self.plotting and self.f_round % 10 == 0:
                    print_confusion_matrix(
                        self.y_test, y_pred, client_id=self.client_id,
                        path_to_out=self.out_dir)
                print(out_1 % (self.client_id, self.f_round,
                               acc, nmi, ami, ari, ran, homo))
                # dumping and retrieving the results
                metrics = {"accuracy": acc,
                           "normalized_mutual_info_score": nmi,
                           "adjusted_mutual_info_score": ami,
                           "adjusted_rand_score": ari,
                           "rand_score": ran,
                           "homogeneity_score": homo}
                result = metrics.copy()
                result['eval_loss'] = loss
                result['client'] = self.client_id
                result['round'] = self.local_iter
                dump_result_dict('client_'+str(self.client_id), result,
                                 path_to_out=self.out_dir)
            if self.id_test is not None:
                pred = {'ID': self.id_test,
                        'label': y_pred}
                dump_pred_dict('pred_client_'+str(self.client_id), pred,
                               path_to_out=self.out_dir)
            result = (loss, len(self.x_test), metrics)
        return result
