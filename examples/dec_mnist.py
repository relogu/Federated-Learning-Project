import os
import pathlib
import numpy as np
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, KLDivLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter

from py.dec.torch.dec import DEC
from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.layers import TruncatedGaussianNoise
from py.dec.torch.utils import cluster_accuracy, get_main_loss, get_mod_loss, get_mod_binary_loss, get_ae_opt, get_linears, target_distribution, get_scaler
from py.datasets.mnist import CachedMNIST
from py.datasets.bmnist import CachedBMNIST
from py.callbacks import ae_training_callback, dec_train_callback, embed_train_callback
from py.util import compute_centroid_np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(
    # cuda, gpu_id, batch_size, pretrain_epochs, finetune_epochs, testing_mode, out_folder,
    #      glw_pretraining, is_tied, ae_main_loss, ae_mod_loss, alpha, input_do, hidden_do, beta,
    #      gaus_noise, ae_opt, lr
         ):
    out_folder = None
    is_tied = True
    testing_mode = False
    gpu_id = 0
    # get configuration dict
    config = {
        'linears': 'dec',
        'f_dim': 10,
        'activation': 'relu',
        'final_activation': 'relu',
        'dropout': 0.0,
        'epochs': 500,
        'n_clusters': 10,
        'ae_batch_size': 256,
        'update_interval': 160,
        'optimizer': 'adam',
        'lr': None,
        'lr_scheduler': True,
        'main_loss': 'mse',
        'mod_loss': 'none',
        'beta': 0.0,
        'corruption': 0.0,
        'noising': 0.0,
        'train_dec': 'yes',
        'alpha': 9,
        'scaler': 'none',
        'binary': False,
    }

    # defining output folder
    if out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(out_folder)
    os.makedirs(path_to_out, exist_ok=True)
    print('Output folder {}'.format(path_to_out))
    # dumping current configuration
    with open(path_to_out/'config.json', 'w') as file:
        json.dump(config, file)
    writer = SummaryWriter(
        logdir=str(str(path_to_out)+'/runs/'),
        flush_secs=5)  # create the TensorBoard object
    # set device for PyTorch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:{}".format(gpu_id)

    # set up loss(es) used in training the SDAE
    if config['binary']:
        if config['mod_loss'] != 'none':
            loss_fn = get_mod_binary_loss(
                name=config['mod_loss'],
                )
            beta = [1.0-config['beta'], config['beta']]
        else:
            loss_fn = [get_main_loss(config['main_loss'])]
            beta = [1.0]
    else:
        if config['mod_loss'] != 'none':
            loss_fn = get_mod_loss(
                name=config['mod_loss'],
                beta=config['beta'],
                main_loss=config['main_loss'],
                device=device,
                )
        else:
            loss_fn = [get_main_loss(config['main_loss'])]
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    # set noising to data  
    noising = None
    if config['noising'] > 0:
        noising = TruncatedGaussianNoise(
            shape=784,
            stddev=config['noising'],
            rate=1.0,
            device=device,
            )
    # set corruption to data
    corruption = None
    if config['corruption'] > 0:
        corruption = config['corruption']
    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        get_linears(config['linears'], 784, config['f_dim']),
        activation=ReLU() if config['activation'] == 'relu' else Sigmoid(),
        final_activation=ReLU() if config['final_activation'] == 'relu' else Sigmoid(),
        dropout=config['dropout'],
        is_tied=is_tied,
    )
    # set learning rate scheduler
    if config['lr_scheduler']:
        scheduler = lambda x: ReduceLROnPlateau(
            x,
            mode='min',
            factor=0.5,
            patience=20,
        )
    else:
        scheduler = None
    # get datasets
    if config['binary']:
        ds_train = CachedBMNIST(
            train=True, device=device, testing_mode=testing_mode
        )  # training dataset
        ds_val = CachedBMNIST(
            train=False, device=device, testing_mode=testing_mode
        )  # evaluation dataset
    else:
        ds_train = CachedMNIST(
            train=True, device=device, testing_mode=testing_mode
        )  # training dataset
        ds_val = CachedMNIST(
            train=False, device=device, testing_mode=testing_mode
        )  # evaluation dataset
    # set dataloaders
    dataloader = DataLoader(
        ds_train,
        batch_size=config['ae_batch_size'],
        shuffle=False,
    )
    validation_loader = DataLoader(
        ds_val,
        batch_size=config['ae_batch_size'],
        shuffle=False,
    )
        
    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        autoencoder.to(device)
        optimizer = get_ae_opt(
            name=config['optimizer'],
            dataset='bmnist' if config['binary'] else 'mnist',
            lr=config['lr'])(autoencoder.parameters())
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        autoencoder.train()
        val_loss = -1
        for epoch in range(config['epochs']):
            running_loss = 0.0
            if scheduler is not None:
                scheduler.step(val_loss)
                
            for i, batch in enumerate(dataloader):
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                batch = batch.to(device)
                input = batch
                
                if noising is not None:
                    input = noising(input)
                if corruption is not None:
                    input = F.dropout(input, corruption)
                output = autoencoder(input)
                
                if config['binary']:
                    losses = [beta*l_fn_i(output, batch) for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch) for l_fn_i in loss_functions]
                loss = sum(losses)/len(loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                # print statistics
                # running_loss += loss.item()
                # print("[%d, %5d] loss: %.3f" % \
                #     (epoch+1, i+1, running_loss / (i+1)))
            running_loss = running_loss / (i+1)
                    
            val_loss = 0.0
            criterion = MSELoss()
            for i, val_batch in enumerate(validation_loader):
                with torch.no_grad():
                    if (
                        isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        val_batch = val_batch[0]
                    val_batch = val_batch.to(device)
                    validation_output = autoencoder(val_batch)
                    loss = criterion(validation_output, val_batch)
                    val_loss += loss.cpu().numpy()
            
            val_loss = ae_training_callback(
                writer,
                'pretraining',
                epoch,
                optimizer.param_groups[0]["lr"],
                running_loss,
                ds_val,
                config,
                device,
                autoencoder)
        torch.save(autoencoder.state_dict(), path_to_out/'pretrain_ae')
        print('Saving features after pretraining.')
        embed_train_callback(
            writer,
            0,
            device,
            (-1, 1, 28, 28),
            config,
            'pretraining',
            (epoch+1),
            ds_val,
            autoencoder)
    if (path_to_out/'finetune_ae').exists():
        print('Skipping finetuning since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    elif noising is not None:
        print("Finetuning stage.")
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
        autoencoder.to(device)
        optimizer = get_ae_opt(
            name=config['optimizer'],
            dataset='bmnist' if config['binary'] else 'mnist',
            lr=config['lr'])(autoencoder.parameters())
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        autoencoder.train()
        val_loss = -1
        for epoch in range(config['epochs']):
            running_loss = 0.0
            if scheduler is not None:
                scheduler.step(val_loss)

            for i, batch in enumerate(dataloader):
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                batch = batch.to(device)
                input = batch
                
                output = autoencoder(input)
                
                if config['binary']:
                    losses = [beta*l_fn_i(output, batch) for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch) for l_fn_i in loss_functions]
                loss = sum(losses)/len(loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                # print statistics
                # running_loss += loss.item()
                # print("[%d, %5d] loss: %.3f" % \
                #     (epoch+1, i+1, running_loss / (i+1)))
            running_loss = running_loss / (i+1)
            
            val_loss = ae_training_callback(
                writer,
                'pretraining',
                epoch,
                optimizer.param_groups[0]["lr"],
                running_loss,
                ds_val,
                config,
                device,
                autoencoder)
        torch.save(autoencoder.state_dict(), path_to_out/'finetune_ae')
        print('Saving features after finetuning.')
        embed_train_callback(
            writer,
            0,
            device,
            (-1, 1, 28, 28),
            config,
            'pretraining',
            (epoch+1),
            ds_val,
            autoencoder)
    
    if config['train_dec'] == 'yes':
        print("DEC stage.")
        dataloader = DataLoader(
            ds_train,
            batch_size=config['ae_batch_size'],# *config['update_interval'],
            shuffle=False,
        )
        autoencoder = autoencoder.to(device)
        model = DEC(cluster_number=config['n_clusters'],
                    hidden_dimension=config['f_dim'],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = get_scaler(config['scaler']) if config['scaler'] != 'none' else None
        kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
        features = []
        actual = []
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch
                actual.append(value)
            batch = batch.to(device, non_blocking=True)
            features.append(model.encoder(batch).detach().cpu())
        actual = torch.cat(actual).long()

        predicted = kmeans.fit_predict(
            scaler.fit_transform(torch.cat(features).numpy()) if scaler is not None else torch.cat(features).numpy()
        )
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
        print("K-Means initial accuracy: %.3f" % (accuracy))

        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))

        cluster_centers = torch.tensor(
            np.array(emp_centroids) if scaler is not None else kmeans.cluster_centers_,
            dtype=torch.float,
            requires_grad=True,
        )
        cluster_centers = cluster_centers.to(device, non_blocking=True)
        with torch.no_grad():
            model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)

        loss_function = KLDivLoss(reduction='sum')
        for epoch in range(20):
            model.train()
            for batch in dataloader:
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                    batch
                ) == 2:
                    batch, _ = batch
                batch = batch.to(device, non_blocking=True)
                output = model(batch)
                soft_labels = output
                target = target_distribution(soft_labels).detach()
                loss = loss_function(output.log(), target) / output.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
            
            predicted_previous = dec_train_callback(
                writer,
                config,
                ds_train,
                model,
                autoencoder,
                device,
                epoch,
                predicted_previous,
            )
            
            embed_train_callback(
                writer,
                0,
                device,
                (-1, 1, 28, 28),
                config,
                'clustering',
                (epoch+1),
                ds_train,
                autoencoder)
        torch.save(model.state_dict(), path_to_out/'dec_model')
    writer.close()


if __name__ == "__main__":
    main()
