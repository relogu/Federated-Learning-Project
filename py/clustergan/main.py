#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:47:12 2021

@author: relogu
"""
import argparse
import pathlib
import os
import sys
import numpy as np
from itertools import chain as ichain

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import py.metrics as my_metrics
import py.dataset_util as data_util
from py.clustergan.util import sample_z, calc_gradient_penalty
from py.clustergan.dense_model import GeneratorDense, DiscriminatorDense, EncoderDense
from py.clustergan.cnn_model import GeneratorCNN, DiscriminatorCNN, EncoderCNN
from py.dumping.plots import plot_lifelines_pred, print_confusion_matrix
from py.dumping.output import dump_pred_dict, dump_result_dict


def get_parser():
    parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
    parser.add_argument("-s", "--dataset", dest="dataset", default='euromds',
                        choices=['mnist', 'euromds'], type=type(''), help="Dataset")
    parser.add_argument("--save_img", dest="save_img",
                        default=False, type=bool, help="Wheather to save images")
    parser.add_argument("-e", "--n_epochs", dest="n_epochs",
                        default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size",
                        default=64, type=int, help="Batch size")
    parser.add_argument("-d", "--latent_dim", dest="latent_dim",
                        default=18, type=int, help="Dimension of latent space")
    parser.add_argument("-n", "--n_clusters", dest="n_clusters",
                        default=6, type=int, help="Dimension of label space")
    parser.add_argument("-l", "--lr", dest="learning_rate",
                        type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-c", "--n_critic", dest="n_critic", type=int,
                        default=5, help="Number of training steps for discriminator per iter")
    parser.add_argument("-w", "--wass_flag", dest="wass_flag",
                        action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-a", "--hardware_acc", dest="cuda_flag", action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("-f", "--folder", dest="out_folder",
                        type=type(str('')), help="Folder to output images")
    parser.add_argument('-g', '--groups', dest='groups', required=False, type=int, choices=[
                        1, 2, 3, 4, 5, 6, 7], default=1, action='store', help='how many groups of variables to use for EUROMDS dataset')
    parser.add_argument('--binary', action='store_true', default=False,
                        help='Use BSN')
    '''
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic activations instead of deterministic [active iff `--binary`]')
    parser.add_argument('--reinforce', action='store_true', default=False,
                        help='Use REINFORCE Estimator instead of Straight Through Estimator [active iff `--binary`]')
    parser.add_argument('--slope-annealing', action='store_true', default=False,
                        help='Use slope annealing trick')'''
    return parser


if __name__ == "__main__":

    # for managing the cpu cores to use
    torch.set_num_threads(8)

    # get parameters
    args = get_parser().parse_args()

    # defining output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    os.makedirs(path_to_out, exist_ok=True)

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = args.learning_rate
    b1 = 0.5
    b2 = 0.9
    decay = 2.5*1e-5
    n_skip_iter = args.n_critic

    # Latent space info
    latent_dim = args.latent_dim
    n_c = args.n_clusters
    betan = 10
    betac = 10
    '''
    # Model, activation type, estimator type
    if args.binary:
        if args.stochastic:
            mode = 'Stochastic'
        else:
            mode = 'Deterministic'
        if args.reinforce:
            estimator = 'REINFORCE'
        else:
            estimator = 'ST'
    # Slope annealing
    if args.slope_annealing:
        get_slope = lambda epoch : 1.0 * (1.005 ** (epoch - 1))
    else:
        get_slope = lambda epoch : 1.0'''
    # BSN
    bsn = args.binary
    print('Using {} network'.format('BSN' if bsn else 'Standard'))

    # Wasserstein+GP metric flag
    wass_metric = args.wass_flag
    print('Using metric {}'.format('Wassestrain' if wass_metric else 'Vanilla'))

    CUDA = True if (torch.cuda.is_available() and args.cuda_flag) else False
    device = torch.device('cuda:0' if CUDA else 'cpu')
    print('Using device {}'.format(device))

    # Data dimensions
    if args.dataset == 'mnist':
        img_size = args.img_size
        channels = 1
        x_shape = (channels, img_size, img_size)

        # Initialize generator and discriminator
        generator = GeneratorCNN(latent_dim, n_c, x_shape)
        encoder = EncoderCNN(latent_dim, n_c)
        discriminator = DiscriminatorCNN(wass_metric=wass_metric)

        # Configure data loader
        #os.makedirs("data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data/mnist/",
                train=True,
                transform=transforms.Compose(
                    [transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        # Test data loader
        testdata = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data/mnist/",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    else:
        groups = ['Genetics', 'CNA', 'GeneGene', 'CytoCyto',
                  'GeneCyto', 'Demographics', 'Clinical']
        # getting the entire dataset
        x = data_util.get_euromds_dataset(groups=groups[:args.groups])
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
        # getting IDs
        ids = data_util.get_euromds_ids()
        n_features = len(x.columns)
        x = np.array(x)
        outcomes = np.array(outcomes)
        ids = np.array(ids)
        # cross-val
        train_idx, test_idx = data_util.split_dataset(
            x=x,
            splits=5,
            fold_n=0)
        # dividing data
        x_train = x[train_idx]
        y_train = y[train_idx]
        id_train = ids[train_idx]
        outcomes_train = outcomes[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        id_test = ids[test_idx]
        outcomes_test = outcomes[test_idx]
        dataloader = DataLoader(
            data_util.PrepareData(x=x_train,
                                  y=y_train,
                                  ids=id_train,
                                  outcomes=outcomes_train),
            batch_size=batch_size)
        testloader = DataLoader(
            data_util.PrepareData(x=x_test,
                                  y=y_test,
                                  ids=id_test,
                                  outcomes=outcomes_test),
            batch_size=batch_size)
        config = {
            'gen_dims': [int(4*n_features), int(3*n_features), int(2*n_features), x.shape[-1]],
            'enc_dims': [int(x.shape[-1]), int(4*n_features), int(3*n_features), int(2*n_features)],
            'disc_dims': [int(x.shape[-1]), int(2*n_features), int(3*n_features), int(4*n_features)]
        }
        generator = GeneratorDense(latent_dim=latent_dim,
                                   n_c=n_c,
                                   gen_dims=config['gen_dims'],
                                   x_shape=x.shape[-1],
                                   use_binary=bsn)
        encoder = EncoderDense(latent_dim=latent_dim,
                               enc_dims=config['enc_dims'],
                               n_c=n_c)
        discriminator = DiscriminatorDense(
            disc_dims=config['disc_dims'], wass_metric=wass_metric)

    torch.autograd.set_detect_anomaly(True)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    if CUDA:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()

    TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())

    optimizer_GE = torch.optim.Adam(
        ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(b1, b2))

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []

    c_zn = []
    c_zc = []
    c_i = []

    # Training loop
    print('\nBegin training session with %i epochs...\n' % (n_epochs))
    for epoch in range(n_epochs):
        # for i, (imgs, itruth_label) in enumerate(dataloader):
        for i, (data) in enumerate(dataloader):

            if args.dataset == 'mnist':
                (imgs, itruth_label) = data
            elif args.dataset == 'euromds':
                (imgs, itruth_label, _, _) = data
            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()

            # Zero gradients for models, resetting at each iteration because they sum up,
            # and we don't want them to pile up between different iterations
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            optimizer_D.zero_grad()
            optimizer_GE.zero_grad()

            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------

            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                      latent_dim=latent_dim,
                                      n_c=n_c,
                                      cuda=CUDA)

            # Generate a batch of images
            gen_imgs = generator(zn, zc)

            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            valid = Variable(TENSOR(gen_imgs.size(0), 1).fill_(
                1.0), requires_grad=False)
            fake = Variable(TENSOR(gen_imgs.size(0), 1).fill_(
                0.0), requires_grad=False)

            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    # ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss # original
                    ge_loss = - torch.mean(D_gen) + betan * \
                        zn_loss + betac * zc_loss  # corrected
                else:
                    # Vanilla GAN loss
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss
                # backpropagate the gradients
                ge_loss.backward(retain_graph=True)
                # computes the new weights
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penaltytorch.autograd.set_detect_anomaly(True) term
                grad_penalty = calc_gradient_penalty(
                    discriminator, real_imgs, gen_imgs, cuda=CUDA)

                # Wasserstein GAN loss w/gradient penalty
                # d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty # original
                d_loss = - torch.mean(D_real) + \
                    torch.mean(D_gen) + grad_penalty  # corrected

            else:
                # Vanilla GAN loss
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(inputs=list(discriminator.parameters()))
            optimizer_D.step()

        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())

        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        if args.dataset == 'mnist':
            test_imgs, test_labels = next(iter(testdata))
            test_imgs = Variable(test_imgs.type(TENSOR))
        elif args.dataset == 'euromds':
            test_imgs, test_labels, test_ids, test_outcomes = next(
                iter(testloader))
            times = test_outcomes[:, 0]
            events = test_outcomes[:, 1]
            test_imgs = Variable(test_imgs.type(TENSOR))

        # Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)

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
        if args.dataset == 'euromds':
            # plotting outcomes on the labels
            plot_lifelines_pred(time=times,
                                event=events,
                                labels=computed_labels,
                                path_to_out=path_to_out)
        if epoch % 10 == 0:  # print confusion matrix
            print_confusion_matrix(
                y=t_label.detach().cpu().numpy(),
                y_pred=computed_labels,
                path_to_out=path_to_out)
        # dumping and retrieving the results
        metrics = {"accuracy": acc,
                   "normalized_mutual_info_score": nmi,
                   "adjusted_mutual_info_score": ami,
                   "adjusted_rand_score": ari,
                   "rand_score": ran,
                   "homogeneity_score": homo}
        result = metrics.copy()

        # Generate sample instances from encoding
        teg_imgs = generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())

        # Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=latent_dim,
                                                 n_c=n_c,
                                                 cuda=CUDA)
        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)

        # Encode sample instances
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)

        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())

        # Save cycled and generated examples!
        if args.save_img:
            r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
            e_zn, e_zc, e_zc_logits = encoder(r_imgs)
            reg_imgs = generator(e_zn, e_zc)
            save_image(reg_imgs.data[:n_samp],
                       path_to_out+('cycle_reg_%06i.png' % (epoch+1)),
                       nrow=n_sqrt_samp, normalize=True)
            save_image(gen_imgs_samp.data[:n_samp],
                       path_to_out+('gen_%06i.png' % (epoch+1)),
                       nrow=n_sqrt_samp, normalize=True)

            # Generate samples for specified classes
            stack_imgs = []
            for idx in range(n_c):
                # Sample specific class
                zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                         latent_dim=latent_dim,
                                                         n_c=n_c,
                                                         fix_class=idx,
                                                         cuda=CUDA)

                # Generate sample instances
                gen_imgs_samp = generator(zn_samp, zc_samp)

                if (len(stack_imgs) == 0):
                    stack_imgs = gen_imgs_samp
                else:
                    stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

            # Save class-specified generated examples!
            save_image(stack_imgs,
                       path_to_out/('gen_classes_%06i.png' % (epoch+1)),
                       nrow=n_c, normalize=True)

        result['img_mse_loss'] = img_mse_loss.item()
        result['lat_mse_loss'] = lat_mse_loss.item()
        result['lat_xe_loss'] = lat_xe_loss.item()
        result['round'] = epoch+1
        dump_result_dict(filename='clustergan',
                         result=result, path_to_out=path_to_out)
        if args.dataset == 'euromds':
            pred = {'ID': test_ids,
                    'label': computed_labels}
            dump_pred_dict(filename='pred', pred=pred,
                           path_to_out=path_to_out)

        print("[Epoch %d/%d] \n"
              "\tModel Losses: [D: %f] [GE: %f]" % (epoch+1,
                                                    n_epochs,
                                                    d_loss.item(),
                                                    ge_loss.item())
              )

        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]" % (img_mse_loss.item(),
                                                               lat_mse_loss.item(),
                                                               lat_xe_loss.item())
              )

        print('Epoch %d/%d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' %
              (epoch+1, n_epochs, acc, nmi, ami, ari, ran, homo))

        g_par = [val.cpu().numpy()
                 for _, val in generator.state_dict().items()]
        d_par = [val.cpu().numpy()
                 for _, val in discriminator.state_dict().items()]
        e_par = [val.cpu().numpy()
                 for _, val in encoder.state_dict().items()]
        parameters = np.concatenate([g_par, d_par, e_par], axis=0)
        np.savez(path_to_out/'clustergan', parameters)
