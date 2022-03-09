from py.parsers.dec_mnist_parser import dec_mnist_parser as get_parser
from py.util import compute_centroid_np
from py.callbacks import ae_training_callback, dec_train_callback, embed_train_callback
from py.datasets.bmnist import CachedBMNIST
from py.datasets.mnist import CachedMNIST
from py.dec.torch.utils import (cluster_accuracy, get_main_loss, get_mod_loss,
                                get_mod_binary_loss, get_opt, get_linears,
                                target_distribution, get_scaler, get_cl_lr,
                                get_cl_batch_size, get_ae_lr)
from py.dec.torch.layers import TruncatedGaussianNoise
from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.dec import DEC
from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import ReLU, Sigmoid, KLDivLoss, MSELoss
import torch.nn.functional as F
import torch
import tensorboard as tb
import tensorflow as tf
import os
import pathlib
import numpy as np
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Parse arguments
    args = get_parser().parse_args()
    path_to_data = args.data_folder
    out_folder = args.out_folder
    is_tied = True
    gpu_id = args.gpu_id
    testing_mode = args.testing_mode
    # Set configuration dict
    config = {
        'linears': args.linears,
        'f_dim': args.hidden_dimensions,
        'activation': args.activation,
        'final_activation': args.final_activation,
        'dropout': args.hidden_dropout,
        'pretrain_epochs': args.pretrain_epochs,
        'finetune_epochs': args.finetune_epochs,
        'n_clusters': args.n_clusters,
        'ae_batch_size': args.ae_batch_size,
        'optimizer': args.opt,
        'ae_lr': args.ae_lr,
        'lr_scheduler': args.lr_sched,
        'main_loss': args.main_loss,
        'mod_loss': args.mod_loss,
        'beta': args.beta,
        'corruption': args.corruption,
        'noising': args.noising,
        'train_dec': args.train_dec,
        'n_init': args.n_init,
        'dec_batch_size': args.dec_batch_size,
        'dec_lr': args.dec_lr,
        'alpha': args.alpha,
        'scaler': args.scaler,
        'binary': args.binary,
    }

    # Define output folder
    if out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(out_folder)
    os.makedirs(path_to_out, exist_ok=True)
    print('Output folder {}'.format(path_to_out))
    # Dump current configuration
    with open(path_to_out/'config.json', 'w') as file:
        json.dump(vars(args), file)
    writer = SummaryWriter(
        logdir=str(str(path_to_out)+'/runs/'),
        flush_secs=5)  # this creates the TensorBoard object
    # Set device for PyTorch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:{}".format(gpu_id)
    # Set up loss(es) used in training the SDAE
    if config['binary']:
        if config['mod_loss'] is not None:
            loss_fn = get_mod_binary_loss(
                name=config['mod_loss'],
            )
            beta = [1.0-config['beta'], config['beta']]
        else:
            loss_fn = [get_main_loss(config['main_loss'])]
            beta = [1.0]
    else:
        if config['mod_loss'] is not None:
            loss_fn = get_mod_loss(
                name=config['mod_loss'],
                beta=config['beta'],
                main_loss=config['main_loss'],
                device=device,
            )
        else:
            loss_fn = [get_main_loss(config['main_loss'])]
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    # Set noising to data
    noising = None
    if config['noising'] > 0:
        noising = TruncatedGaussianNoise(
            shape=784,
            stddev=config['noising'],
            rate=1.0,
            device=device,
        )
    # Set corruption to data
    corruption = None
    if config['corruption'] > 0:
        corruption = config['corruption']
    # Set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        get_linears(config['linears'], 784, config['f_dim']),
        activation=ReLU() if config['activation'] == 'relu' else Sigmoid(),
        final_activation=ReLU(
        ) if config['final_activation'] == 'relu' else Sigmoid(),
        dropout=config['dropout'],
        is_tied=is_tied,
    )
    # Set learning rate scheduler
    if config['lr_scheduler']:
        def scheduler(x): return ReduceLROnPlateau(
            x,
            mode='min',
            factor=0.5,
            patience=20,
        )
    else:
        scheduler = None
    # Get datasets
    if config['binary']:
        ds_train = CachedBMNIST(
            path=path_to_data, train=True, device=device, testing_mode=testing_mode
        )  # training dataset
        ds_val = CachedBMNIST(
            path=path_to_data, train=False, device=device, testing_mode=testing_mode
        )  # evaluation dataset
    else:
        ds_train = CachedMNIST(
            path=path_to_data, train=True, device=device, testing_mode=testing_mode
        )  # training dataset
        ds_val = CachedMNIST(
            path=path_to_data, train=False, device=device, testing_mode=testing_mode
        )  # evaluation dataset
    # Set dataloaders
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
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                dataset='bmnist' if config['binary'] else 'mnist',
                linears=config['linears'],
                opt=config['optimizer']),
        )(autoencoder.parameters())
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        autoencoder.train()
        val_loss = -1
        for epoch in range(config['pretrain_epochs']):
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
                    losses = [beta*l_fn_i(output, batch)
                              for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch)
                              for l_fn_i in loss_functions]
                loss = sum(losses)/len(loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
            running_loss = running_loss / (i+1)

            val_loss = 0.0
            criterion = MSELoss()
            for i, val_batch in enumerate(validation_loader):
                with torch.no_grad():
                    if (
                        isinstance(val_batch, tuple) or isinstance(
                            val_batch, list)
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
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                dataset='bmnist' if config['binary'] else 'mnist',
                linears=config['linears'],
                opt=config['optimizer']),
        )(autoencoder.parameters())
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        autoencoder.train()
        val_loss = -1
        for epoch in range(config['finetune_epochs']):
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
                    losses = [beta*l_fn_i(output, batch)
                              for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch)
                              for l_fn_i in loss_functions]
                loss = sum(losses)/len(loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
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
            batch_size=get_cl_batch_size(
                linears='dec',
                dataset='bmnist' if config['binary'] else 'mnist',
                opt=config['optimizer']) if config['dec_batch_size'] is not None else config['dec_batch_size'],
            shuffle=False,
        )
        autoencoder = autoencoder.to(device)
        model = DEC(cluster_number=config['n_clusters'],
                    hidden_dimension=config['f_dim'],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])
        model = model.to(device)
        optimizer = get_opt(
            name=config['optimizer'],
            lr=config['dec_lr'] if config['dec_lr'] is not None else get_cl_lr(
                linears='dec',
                dataset='bmnist' if config['binary'] else 'mnist',
                opt=config['optimizer']),
        )(model.parameters())
        scaler = get_scaler(
            config['scaler']) if config['scaler'] != 'none' else None
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
            scaler.fit_transform(torch.cat(features).numpy(
            )) if scaler is not None else torch.cat(features).numpy()
        )
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
        print("K-Means initial accuracy: %.3f" % (accuracy))

        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(
                torch.cat(features).numpy()[idx, :]))

        cluster_centers = torch.tensor(
            np.array(
                emp_centroids) if scaler is not None else kmeans.cluster_centers_,
            dtype=torch.float,
            requires_grad=True,
        )
        cluster_centers = cluster_centers.to(device, non_blocking=True)
        with torch.no_grad():
            model.state_dict()["assignment.cluster_centers"].copy_(
                cluster_centers)

        loss_function = KLDivLoss(reduction='sum')
        for epoch in range(config['dec_epochs']):
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
