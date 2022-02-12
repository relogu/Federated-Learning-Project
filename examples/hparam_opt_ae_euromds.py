from functools import partial
import os
import pathlib
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, KLDivLoss
from torch.nn.modules.loss import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.cluster import KMeans

from py.dec.dec_torch.dec import DEC
from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.dec.layers.torch import TruncatedGaussianNoise
from py.datasets.euromds import CachedEUROMDS
from py.dec.dec_torch.utils import get_ae_opt, get_main_loss, get_mod_binary_loss, get_scaler, cluster_accuracy, target_distribution
from py.util import compute_centroid_np

def train_ae(
    config: Dict,
    scheduler: Any = None,
    device: str = 'cpu',
) -> None:
    """
    TODO

    :param config: TODO
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param device: TODO
    :return: None
    """
    ## Instantiate DataLoaders
    ds_train = CachedEUROMDS(
        exclude_cols=['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
        groups=['Genetics', 'CNA'],
        path_to_data=pathlib.Path('~/Federated-Learning-Project/data/euromds'),
        fill_nans=2044,
        get_hdp=True,
        get_outcomes=True,
        get_ids=True,
        verbose=True,
        device=device,
    )  # training dataset
    ds_val = ds_train # evaluation dataset
    dataloader = DataLoader(
        ds_train,
        batch_size=config['batch_size'],
        shuffle=False,
    )
    validation_loader = DataLoader(
        ds_val,
        batch_size=config['batch_size'],
        shuffle=False,
    )
    ## SDAE Training Loop
    # set up loss(es) used in training the SDAE
    if config['mod_loss'] != 'none':
        loss_fn = get_mod_binary_loss(
            name=config['mod_loss'],
            # beta=beta,
            # main_loss=ae_main_loss,
            # device=device,
            )
    else:
        loss_fn = [get_main_loss(config['main_loss'])]
    
    noising = None
    if config['noising'] > 0:
        noising = TruncatedGaussianNoise(
            shape=ds_train.n_features,
            stddev=config['noising'],
            rate=1.0,
            device=device,
            )
        
    corruption = None
    if config['corruption'] > 0:
        corruption = config['corruption']
        
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    
    def get_linears(name, f_dim):
        linears_dict = {
            'dec': [ds_train.n_features, 500, 500, 2000, f_dim],
            'google': [ds_train.n_features, 1000, 500, 250, f_dim],
            'curves': [ds_train.n_features, 400, 200, 100, 50, 25, 6],
        }
        return linears_dict[name]
    
    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        get_linears(config['linears'], config['f_dim']),
        activation=config['activation'],
        final_activation=config['final_activation'],
        dropout=config['dropout'],
        is_tied=True,
    )
    if torch.cuda.device_count() > 1:
        autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.to(device)
    optimizer = get_ae_opt(config['optimizer'], config['lr'])(autoencoder.parameters())
    scheduler = scheduler(optimizer)
    
    autoencoder.train()
    last_loss = -1
    for epoch in range(config['epochs']):
        running_loss = 0.0
        epoch_steps = 0
        if scheduler is not None:
            scheduler.step(last_loss)
            
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
            
            losses = [l_fn_i(output, batch) for l_fn_i in loss_functions]
            loss = sum(losses)/len(loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
                
        val_loss = 0.0
        val_steps = 0
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
                val_steps += 1

        last_loss = (val_loss / val_steps)
        tune.report(loss=last_loss)
        
    if noising is not None:
        for epoch in range(config['epochs']):
            running_loss = 0.0
            epoch_steps = 0
            if scheduler is not None:
                scheduler.step(last_loss)

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

                losses = [l_fn_i(output, batch) for l_fn_i in loss_functions]
                loss = sum(losses)/len(loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0

            val_loss = 0.0
            val_steps = 0
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
                    val_steps += 1
                    
            last_loss = (val_loss / val_steps)
            tune.report(loss=last_loss)
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "SDAE_checkpoint")
        torch.save((autoencoder.state_dict(), optimizer.state_dict()), path)
        
    print("Finished SDAE Training")
    
    if config['train_dec'] == 'yes':
        model = DEC(cluster_number=10,
                    hidden_dimension=get_linears(config['linears'], config['f_dim'])[-1],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])

        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        scaler = get_scaler(config['scaler']) if config['scaler'] != 'none' else None
        kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
        features = []
        actual = []
        # form initial cluster centres
        for index, batch in enumerate(dataloader):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
                actual.append(value)
            batch = batch.to(device, non_blocking=True)
            features.append(model.encoder(batch).detach().cpu())
        actual = torch.cat(actual).long()

        predicted = kmeans.fit_predict(
            scaler.fit_transform(torch.cat(features).numpy()) if scaler is not None else torch.cat(features).numpy()
        )
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
        tune.report(accuracy=accuracy)

        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))

        # true_centroids = []
        # for i in np.unique(actual.cpu().numpy()):
        #     idx = (actual.cpu().numpy() == i)
        #     true_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))

        cluster_centers = torch.tensor(
            np.array(emp_centroids) if config['use_emp_centroids'] == 'yes' else kmeans.cluster_centers_,# np.array(true_centroids)
            dtype=torch.float,
            requires_grad=True,
        )
        cluster_centers = cluster_centers.to(device, non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)

        loss_function = KLDivLoss(size_average=False)
        delta_label = None
        for epoch in range(20):
            # predicted_previous, accuracy = assign_cluster_centers(
            #     dataset=dataset,
            #     model=model,
            #     batch_size=batch_size,
            #     collate_fn=collate_fn,
            #     device=device,
            # )
            # old_model = copy.deepcopy(model)
            model.train()
            for index, batch in enumerate(dataloader):
                # if index % 140:
                #     old_model = copy.deepcopy(model)
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                    batch
                ) == 2:
                    batch, _ = batch  # if we have a prediction label, strip it away
                batch = batch.to(device, non_blocking=True)
                output = model(batch)
                soft_labels = output
                # soft_labels = old_model(batch)
                target = target_distribution(soft_labels).detach()
                loss = loss_function(output.log(), target) / output.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)

            features = []
            actual = []
            model.eval()
            for batch in dataloader:
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    batch, value = batch  # unpack if we have a prediction label
                    actual.append(value)
                batch = batch.to(device, non_blocking=True)
                features.append(
                    model(batch).detach().cpu()
                )  # move to the CPU to prevent out of memory on the GPU
            predicted, actual = torch.cat(features).max(1)[1], torch.cat(actual).long()

            delta_label = (
                float((predicted != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
            )

            tune.report(delta_label=delta_label)

            predicted_previous = predicted
            _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())

            tune.report(accuracy=accuracy)
        tune.report(loss=last_loss, accuracy=accuracy)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "DEC_checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        print("Finished DEC Training")


def main(num_samples=1, max_num_epochs=500, gpus_per_trial=1):
    
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda:0"

    config = {
        'linears': 'dec',# tune.grid_search(['dec', 'google', 'curves']),
        'f_dim': 10,# tune.choice([6,9,10,20,30]),# tune.randint(2, 30),# 10,# tune.grid_search([9,10,11,12,13]),# tune.grid_search([10, 30]),
        'activation': ReLU(),# tune.grid_search([ReLU(), Sigmoid()]),
        'final_activation': Sigmoid(),# tune.grid_search([ReLU(), Sigmoid()]),
        'dropout': tune.uniform(0.0, 0.5),# tune.grid_search([0.0, 0.2, 0.4, 0.5]),
        'epochs': max_num_epochs,
        'batch_size': 256,
        'optimizer': 'yogi',# tune.grid_search(['adam', 'yogi']),# tune.grid_search(['adam', 'yogi', 'sgd']),
        'lr': tune.loguniform(1e-5, 1e-1),
        'main_loss': 'mse',# tune.grid_search(['mse', 'bce-wl']),
        'mod_loss': 'none',# tune.grid_search(['none', 'gausk1', 'gausk3']),# tune.grid_search(['mix', 'gausk1', 'gausk3']),
        'beta': 0.0,# tune.grid_search([0.1, 0.2]),
        'corruption': tune.uniform(0.0, 0.5),# tune.grid_search([0.0, 0.1, 0.2, 0.3,]),
        'noising': 0.0,# tune.grid_search([0.0, 0.1]),
        'train_dec': 'yes',
        'alpha': 9,# tune.grid_search([1, 9]),
        'scaler': 'normal-l2',# tune.grid_search(['standard', 'normal-l1', 'normal-l2', 'none']),
        'use_emp_centroids': 'yes',#tune.grid_search(['yes', 'no']),
    }
    
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "delta_label", "training_iteration"]
        )
    
    lambda_scheduler = lambda x: ReduceLROnPlateau(
        x,
        mode='min',
        factor=0.5,
        patience=20,
    )
    
    # bayesopt = BayesOptSearch(metric="loss", mode="min")
                
    result = tune.run(
        partial(train_ae, scheduler=lambda_scheduler, device=device),
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        # scheduler=scheduler,
        # search_alg=bayesopt,
        progress_reporter=reporter,
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best SDAE trial config: {}".format(best_trial.config))
    print("Best SDAE trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best DEC trial config: {}".format(best_trial.config))
    print("Best DEC trial final accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
    
