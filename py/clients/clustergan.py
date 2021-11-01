#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:55:15 2021

@author: relogu
"""
import os
from itertools import chain as ichain
from typing import OrderedDict, Dict, Callable, Union
from pathlib import Path

import numpy as np
import torch
from py.clustergan.dense_model import DiscriminatorDense, EncoderDense, GeneratorDense
from py.clustergan.cnn_model import DiscriminatorCNN, EncoderCNN, GeneratorCNN
from py.clustergan.util import calc_gradient_penalty, sample_z

from flwr.client import NumPyClient
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import py.metrics as my_metrics
from py.dataset_util import split_dataset, PrepareData, PrepareDataSimple
from py.util import check_weights_dict
from py.dumping.output import dump_pred_dict, dump_result_dict
from py.dumping.plots import print_confusion_matrix, plot_lifelines_pred


class ClusterGANClient(NumPyClient):

    def __init__(self,
                 client_id,  # id of the client
                 config: Dict = None,  # configuration dictionary
                 get_data_fn: Callable = None, # fn for getting dataset
                 hardw_acc_flag: bool = False,
                 output_folder: Union[Path, str] = None,
                 seed: int = 51550  # seed for random gen
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
        
        # Check for hardware acceleration
        self.cuda = True if torch.cuda.is_available() and hardw_acc_flag else False
        self.device = torch.device(
            'cuda:0' if self.cuda else 'cpu')
        print('Using device {}'.format(self.device))
        torch.autograd.set_detect_anomaly(True)

        # Loss functions
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
        # TODO: get dataset
        self.train, self.test = get_data_fn(client_id, config['dataset'])
        self.trainloader = DataLoader(
            PrepareDataSimple(self.train),
            batch_size=self.batch_size)
        self.testloader = DataLoader(
            PrepareDataSimple(self.test),
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
            self.out_dir = Path(output_folder)
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
        print('Client %d, FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami '
              '%.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' % \
                  (self.client_id, self.f_epoch,
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
