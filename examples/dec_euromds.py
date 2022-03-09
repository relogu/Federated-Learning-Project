from py.callbacks import ae_training_callback, dec_train_callback, embed_train_callback
from py.util import get_square_image_repr, compute_centroid_np
from py.datasets.euromds import CachedEUROMDS
from py.dec.torch.utils import (cluster_accuracy, get_main_loss, get_mod_binary_loss,
                                get_opt, get_linears, get_scaler, target_distribution,
                                get_ae_lr, get_cl_lr, get_cl_batch_size)
from py.dec.torch.layers import TruncatedGaussianNoise
from py.dec.torch.sdae import StackedDenoisingAutoEncoder
from py.dec.torch.dec import DEC
from py.parsers.dec_euromds_parser import dec_euromds_parser as get_parser
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
    }
    # Define output folder
    if out_folder is None:
        path_to_out = pathlib.Path(
            __file__).parent.parent.absolute()/'euromds_cl_best_modloss'
    else:
        path_to_out = pathlib.Path(out_folder)
    os.makedirs(path_to_out, exist_ok=True)
    print('Output folder {}'.format(path_to_out))
    # Define data folder
    if path_to_data is None:
        path_to_data = pathlib.Path(
            __file__).parent.parent.absolute()/'data/euromds'
    else:
        path_to_data = pathlib.Path(path_to_data)
    print('Data folder {}'.format(path_to_data))
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
    if config['mod_loss'] is not None:
        loss_fn = get_mod_binary_loss(
            name=config['mod_loss'],
        )
        beta = [1.0-config['beta'], config['beta']]
    else:
        loss_fn = [get_main_loss(config['main_loss'])]
        beta = [1.0]
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    # Get dataset
    ds_train = CachedEUROMDS(
        exclude_cols=args.ex_col,  # ['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
        groups=args.groups,  # ['Genetics', 'CNA'],
        path_to_data=path_to_data,
        fill_nans=args.fill_nans,  # 2044,
        get_hdp=True,
        get_outcomes=True,
        get_ids=True,
        verbose=True,
        device=device,
    )  # training dataset
    ds_val = ds_train
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
    img_repr = get_square_image_repr(ds_train.n_features)
    print("Square image representation for {} features is (x,y,add): {}".format(
        ds_train.n_features, img_repr))
    additions = img_repr[2]
    img_repr = (-1, 1, img_repr[0], img_repr[1])
    # Set noising to data
    noising = None
    if config['noising'] > 0:
        noising = TruncatedGaussianNoise(
            shape=ds_train.n_features,
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
        get_linears(config['linears'], ds_train.n_features, config['f_dim']),
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

    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        autoencoder.to(device)
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                dataset='euromds',
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

                losses = [beta*l_fn_i(output, batch)
                          for beta, l_fn_i in zip(beta, loss_functions)]
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
            additions,
            device,
            img_repr,
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
            name=config['optimizer'],
            dataset='euromds',
            linears=config['linears'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                dataset='euromds',
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

                losses = [beta*l_fn_i(output, batch)
                          for beta, l_fn_i in zip(beta, loss_functions)]
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
            additions,
            device,
            img_repr,
            config,
            'pretraining',
            (epoch+1),
            ds_val,
            autoencoder)

    if config['train_dec']:
        print("DEC stage.")
        dataloader = DataLoader(
            ds_train,
            batch_size=config['dec_batch_size'] if config['dec_batch_size'] is not None else get_cl_batch_size(
                linears='dec',
                dataset='euromds',
                opt=config['optimizer']),
            shuffle=False,
        )
        autoencoder = autoencoder.to(device)
        model = DEC(cluster_number=config['n_clusters'],
                    hidden_dimension=config['f_dim'],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])
        model = model.to(device)
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['dec_lr'] if config['dec_lr'] is not None else get_cl_lr(
                linears=config['linears'],
                dataset='euromds',
                opt=config['optimizer']),
        )(model.parameters())
        scaler = get_scaler(
            config['scaler']) if config['scaler'] != 'none' else None
        kmeans = KMeans(
            n_clusters=model.cluster_number,
            n_init=config['n_init'])
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
                ds_val,
                model,
                autoencoder,
                device,
                epoch,
                predicted_previous,
            )

            embed_train_callback(
                writer,
                additions,
                device,
                img_repr,
                config,
                'clustering',
                (epoch+1),
                ds_val,
                autoencoder)
        torch.save(model.state_dict(), path_to_out/'dec_model')
    writer.close()


if __name__ == "__main__":
    main()
