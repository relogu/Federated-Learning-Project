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
# Set to solve an issue
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Auto-generated
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# Restrict the number of available GPUs to the first
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()
    path_to_data = args.data_folder
    out_folder = args.out_folder
    # Set the AE to be tied
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
    # Create the TensorBoard object for output
    writer = SummaryWriter(
        logdir=str(str(path_to_out)+'/runs/'),
        flush_secs=5)
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
    # Set learning rate scheduler, if asked
    if config['lr_scheduler']:
        def scheduler(x): return ReduceLROnPlateau(
            x,
            mode='min',
            factor=0.5,
            patience=20,
        )
    else:
        scheduler = None
    # Get datasets, discriminating between MNIST and BMNIST
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
    ## PRETRAINING STAGE: training the autoencoder initialized with standard init
    ## This stage is skipped if in the output folder, trained weights are present
    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        # Send net to device
        autoencoder.to(device)
        # Set up the optimizer from the parameters
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                # FIXME: warning about nested conditional expression
                dataset='bmnist' if config['binary'] else 'mnist',
                linears=config['linears'],
                opt=config['optimizer']),
        )(autoencoder.parameters())
        # Set lr scheduler, if asked
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        # Set net in train mode
        autoencoder.train()
        # Init validation loss
        val_loss = -1
        # Training loop
        for epoch in range(config['pretrain_epochs']):
            # Init train running loss
            running_loss = 0.0
            # Step with lr scheduler, if asked
            if scheduler is not None:
                scheduler.step(val_loss)
            # Training batch loop
            for i, batch in enumerate(dataloader):
                # Get the important part of a batch for training 
                # and put it into "input"
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                batch = batch.to(device)
                input = batch
                # Pass through noising layer, if asked
                if noising is not None:
                    input = noising(input)
                # Pass through corruption layer, if asked
                if corruption is not None:
                    input = F.dropout(input, corruption)
                # Pass through entire AE
                output = autoencoder(input)
                # Compute the losses chosen
                if config['binary']:
                    losses = [beta*l_fn_i(output, batch)
                              for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch)
                              for l_fn_i in loss_functions]
                # Average losses
                loss = sum(losses)/len(loss_fn)
                # Reset optimizer
                optimizer.zero_grad()
                # Backward propagation
                loss.backward()
                # Optimize weights
                optimizer.step(closure=None)
            # Update running loss
            running_loss = running_loss / (i+1)
            # Callback for output every epoch to Tensorboard,
            # inside here validation is performed
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
        # Save state dict of the final weights after pretraining
        torch.save(autoencoder.state_dict(), path_to_out/'pretrain_ae')
        print('Saving features after pretraining.')
        # Callback for output the embeddings after pretraining to Tensorboard
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
    ## FINETUNING STAGE: training the autoencoder initialized with the pretrained weights
    ## This stage is skipped if in the output folder, trained weights are present, or if
    ## finetuning is not requested (it's requested and useful only when noising input during
    ## pretraining)
    if (path_to_out/'finetune_ae').exists():
        print('Skipping finetuning since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    elif noising is not None:
        print("Finetuning stage.")
        # Load pretrained parameters (these must exist)
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
        # Send net to device
        autoencoder.to(device)
        # Set up the optimizer from the parameters
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['ae_lr'] if config['ae_lr'] is not None else get_ae_lr(
                # FIXME: warning about nested conditional expression
                dataset='bmnist' if config['binary'] else 'mnist',
                linears=config['linears'],
                opt=config['optimizer']),
        )(autoencoder.parameters())
        # Set lr scheduler, if asked
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        # Set net in train mode
        autoencoder.train()
        # Init validation loss
        val_loss = -1
        # Training loop
        for epoch in range(config['finetune_epochs']):
            # Init train running loss
            running_loss = 0.0
            # Step with lr scheduler, if asked
            if scheduler is not None:
                scheduler.step(val_loss)
            # Training batch loop
            for i, batch in enumerate(dataloader):
                # Get the important part of a batch for training 
                # and put it into "input"
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                batch = batch.to(device)
                input = batch
                # Pass through entire AE
                output = autoencoder(input)
                # Compute the losses chosen
                if config['binary']:
                    losses = [beta*l_fn_i(output, batch)
                              for beta, l_fn_i in zip(beta, loss_functions)]
                else:
                    losses = [l_fn_i(output, batch)
                              for l_fn_i in loss_functions]
                # Average losses
                loss = sum(losses)/len(loss_fn)
                # Reset optimizer
                optimizer.zero_grad()
                # Backward propagation
                loss.backward()
                # Optimize weights
                optimizer.step(closure=None)
            # Update running loss
            running_loss = running_loss / (i+1)
            # Callback for output every epoch to Tensorboard,
            # inside here validation is performed
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
        # Save state dict of the final weights after pretraining
        torch.save(autoencoder.state_dict(), path_to_out/'finetune_ae')
        print('Saving features after finetuning.')
        # Callback for output the embeddings after pretraining to Tensorboard
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
    ## CLUSTERING STAGE: training the autoencoder stacked with the clustering layer
    if config['train_dec'] == 'yes':
        print("DEC stage.")
        # Reset dataloader
        dataloader = DataLoader(
            ds_train,
            batch_size=get_cl_batch_size(
                linears='dec',
                # FIXME: warning about nested conditional expression
                dataset='bmnist' if config['binary'] else 'mnist',
                opt=config['optimizer']) if config['dec_batch_size'] is not None else config['dec_batch_size'],
            shuffle=False,
        )
        # Send net to device
        autoencoder = autoencoder.to(device)
        # Set up DEC model
        model = DEC(cluster_number=config['n_clusters'],
                    hidden_dimension=config['f_dim'],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])
        # Send DEC model to device
        model = model.to(device)
        # Set up the optimizer from the parameters
        optimizer = get_opt(
            opt=config['optimizer'],
            lr=config['dec_lr'] if config['dec_lr'] is not None else get_cl_lr(
                linears='dec',
                # FIXME: warning about nested conditional expression
                dataset='bmnist' if config['binary'] else 'mnist',
                opt=config['optimizer']),
        )(model.parameters())
        # Set up the scaler if asked
        scaler = get_scaler(
            config['scaler']) if config['scaler'] != 'none' else None
        # Set up KMeans algorithm
        kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
        # Loop for KMeans
        features = []
        actual = []
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch
                actual.append(value)
            batch = batch.to(device, non_blocking=True)
            features.append(model.encoder(batch).detach().cpu())
        # Retrieve true labels (in this case they exist)
        actual = torch.cat(actual).long()
        # Predict labels via KMeans
        predicted = kmeans.fit_predict(
            scaler.fit_transform(torch.cat(features).numpy(
            )) if scaler is not None else torch.cat(features).numpy()
        )
        # Pass predicted labels to be new "previous_predicted" for computing
        # delta of changin in next steps
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
        print("K-Means initial accuracy: %.3f" % (accuracy))
        # Computing empirical clusters' centroids given the predicted labels
        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(
                torch.cat(features).numpy()[idx, :]))
        # Set up the torch tensor containing initial centroids to be passed to
        # the clustering layer
        cluster_centers = torch.tensor(
            np.array(
                emp_centroids) if scaler is not None else kmeans.cluster_centers_,
            dtype=torch.float,
            requires_grad=True,
        )
        # Send centroids' tensor to device
        cluster_centers = cluster_centers.to(device, non_blocking=True)
        # Pass centroids to clustering layer
        with torch.no_grad():
            model.state_dict()["assignment.cluster_centers"].copy_(
                cluster_centers)
        # Intantiate the Kullback-Leibler Divergence loss for clustering stage
        loss_function = KLDivLoss(reduction='sum')
        # Training loop
        for epoch in range(config['dec_epochs']):
            # Set DEC model to be in train mode
            model.train()
            # Batch training loop
            for batch in dataloader:
                # Get the important part of a batch for training 
                # and put it into "batch"
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                    batch
                ) == 2:
                    batch, _ = batch
                batch = batch.to(device, non_blocking=True)
                # Pass through DEC model 
                output = model(batch)
                # Get soft label assignments
                soft_labels = output
                # Get target of the clustering optimization given the soft labels
                target = target_distribution(soft_labels).detach()
                # Compute loss
                loss = loss_function(output.log(), target) / output.shape[0]
                # Reset optimizer
                optimizer.zero_grad()
                # Backward propagation
                loss.backward()
                # Optimize weights
                optimizer.step(closure=None)
            # Callback for output every epoch to Tensorboard,
            # inside here some sort of validation is performed.
            # Especially, the clustering metrics chosen for this
            # work are computed.
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
            # Callback for output the embeddings at every epoch to Tensorboard
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
        # Save state dict of the final parameters of the model
        torch.save(model.state_dict(), path_to_out/'dec_model')
    # Close the Summary Writer
    writer.close()
