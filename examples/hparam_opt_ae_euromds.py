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
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from py.dec.dec_torch.dec import DEC
from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.dec.layers.torch import TruncatedGaussianNoise
from py.datasets.euromds import CachedEUROMDS
from py.dec.dec_torch.utils import get_ae_opt, get_main_loss, get_mod_binary_loss, get_scaler, cluster_accuracy, target_distribution, get_linears
from py.util import compute_centroid_np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
    # Instantiate DataLoaders
    ds_train = CachedEUROMDS(
        exclude_cols=['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
        groups=['Genetics', 'CNA'],
        path_to_data=pathlib.Path('~/Federated-Learning-Project/data/euromds'),
        fill_nans=2044,
        get_hdp=True,
        get_outcomes=True,
        get_ids=True,
        verbose=False,
        device=device,
    )  # training dataset
    ds_val = ds_train  # evaluation dataset
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
    # SDAE Training Loop
    # set up loss(es) used in training the SDAE
    if config['mod_loss'] != 'none':
        loss_fn = get_mod_binary_loss(
            name=config['mod_loss'],
            # beta=beta,
            # main_loss=ae_main_loss,
            # device=device,
        )
        beta = [1.0-config['beta'], config['beta']]
    else:
        loss_fn = [get_main_loss(config['main_loss'])]
        beta = [1.0]

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

    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        get_linears(config['linears'], ds_train.n_features, config['f_dim']),
        activation=config['activation'],
        final_activation=config['final_activation'],
        dropout=config['dropout'],
        is_tied=True,
    )
    if torch.cuda.device_count() > 1:
        autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.to(device)
    optimizer = get_ae_opt(config['optimizer'], config['lr'])(
        autoencoder.parameters())
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

            losses = [beta*l_fn_i(output, batch)
                      for beta, l_fn_i in zip(beta, loss_functions)]
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

        last_loss = (val_loss / (i+1))
        tune.report(ae_loss=last_loss)

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "SDAE_pretraining_checkpoint")
        torch.save((autoencoder.state_dict(), optimizer.state_dict()), path)

    # N.B.: corruptions does not need finetuning
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

                losses = [beta*l_fn_i(output, batch)
                          for beta, l_fn_i in zip(beta, loss_functions)]
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
            last_loss = val_loss / (i+1)
            tune.report(ae_loss=last_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "SDAE_finetuning_checkpoint")
            torch.save((autoencoder.state_dict(),
                       optimizer.state_dict()), path)

    print("Finished SDAE Training")

    if config['train_dec'] == 'yes':
        dataloader = DataLoader(
            ds_train,
            # change for including update interval procedure
            batch_size=int(config['ae_batch_size']*config['update_interval']),
            shuffle=True,
        )
        model = DEC(cluster_number=config['n_clusters'],
                    hidden_dimension=config['f_dim'],
                    encoder=autoencoder.encoder,
                    alpha=config['alpha'])

        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        scaler = get_scaler(
            config['scaler']) if config['scaler'] != 'none' else None
        kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
        features = []
        actual = []
        # form initial cluster centres
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
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
        tune.report(accuracy=accuracy)

        emp_centroids = []
        for i in np.unique(predicted):
            idx = (predicted == i)
            emp_centroids.append(compute_centroid_np(
                torch.cat(features).numpy()[idx, :]))

        cluster_centers = torch.tensor(
            # np.array(true_centroids)
            np.array(
                emp_centroids) if config['use_emp_centroids'] == 'yes' else kmeans.cluster_centers_,
            dtype=torch.float,
            requires_grad=True,
        )
        cluster_centers = cluster_centers.to(device, non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            model.state_dict()["assignment.cluster_centers"].copy_(
                cluster_centers)

        loss_function = KLDivLoss(size_average=False)
        delta_label = None
        for epoch in range(20):
            model.train()
            for batch in dataloader:
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                    batch
                ) == 2:  # if we have a prediction label, strip it away
                    batch, _ = batch
                batch = batch.to(device, non_blocking=True)
                output = model(batch)
                soft_labels = output
                target = target_distribution(soft_labels).detach()
                loss = loss_function(output.log(), target) / output.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)

            cl_recon = 0.0
            criterion = MSELoss()
            data = []
            r_data = []
            features = []
            prob_labels = []
            r_prob_labels = []
            actual = []
            model.eval()
            autoencoder.eval()
            for i, batch in enumerate(validation_loader):
                with torch.no_grad():
                    if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                        batch, value = batch  # unpack if we have a prediction label
                        actual.append(value.cpu())
                        data.append(batch.cpu())
                    batch = batch.to(device, non_blocking=True)
                    r_batch = autoencoder(batch)
                    f_batch = autoencoder.encoder(batch)
                    features.append(f_batch.cpu())
                    r_data.append(r_batch.cpu())
                    loss = criterion(r_batch, batch)
                    cl_recon += loss.cpu().numpy()
                    prob_labels.append(model(batch).cpu())
                    r_prob_labels.append(model(r_batch).cpu())

            cl_recon = (cl_recon / (i+1))
            predicted = torch.cat(prob_labels).max(1)[1]
            r_predicted = torch.cat(r_prob_labels).max(1)[1]
            actual = torch.cat(actual).long()
            data = torch.cat(data).numpy()
            r_data = torch.cat(r_data).numpy()
            features = torch.cat(features).numpy()

            delta_label = (
                float((predicted != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
            )

            cos_sil_score = silhouette_score(
                X=data,
                labels=predicted,
                metric='cosine')

            eucl_sil_score = silhouette_score(
                X=features,
                labels=predicted,
                metric='euclidean')

            data_calinski_harabasz = calinski_harabasz_score(
                X=data,
                labels=predicted)

            feat_calinski_harabasz = calinski_harabasz_score(
                X=features,
                labels=predicted)

            predicted_previous = predicted
            _, accuracy = cluster_accuracy(predicted.numpy(), actual.numpy())
            _, cycle_accuracy = cluster_accuracy(
                r_predicted.numpy(), predicted.numpy())

            tune.report(
                accuracy=accuracy,
                cycle_accuracy=cycle_accuracy,
                cl_recon=cl_recon,
                delta_label=delta_label,
                ae_loss=last_loss,
                cos_sil_score=cos_sil_score,
                eucl_sil_score=eucl_sil_score,
                data_calinski_harabasz=data_calinski_harabasz,
                feat_calinski_harabasz=feat_calinski_harabasz,
            )
            # break loop procedure
            if delta_label <= 0.001:
                break

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "DEC_checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        print("Finished DEC Training")


def main(num_samples=1, max_num_epochs=150, gpus_per_trial=0.5):

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"

    config = {
        # tune.grid_search(['dec', 'google']),# , 'curves']),
        'linears': 'dec',
        # tune.grid_search([2,3,4,5,6,7,8,9,10,11,12,13,14]),# tune.choice([6,9,10,20,30]),# tune.randint(2, 30),# 10,# tune.grid_search([9,10,11,12,13]),# tune.grid_search([10, 30]),
        'f_dim': tune.grid_search([9, 10]),
        'activation': ReLU(),  # tune.grid_search([ReLU(), Sigmoid()]),
        # tune.grid_search([ReLU(), Sigmoid()]),
        'final_activation': Sigmoid(),
        # tune.grid_search([0.0, 0.25, 0.5]),# tune.uniform(0.0, 0.5),
        'dropout': 0.0,
        'epochs': max_num_epochs,
        'n_clusters': tune.grid_search([6, 7, 8, 9, 10]),
        'ae_batch_size': 8,  # tune.grid_search([8,16,32]),# 256,
        'update_interval': 50,  # tune.grid_search([20, 50, 100]),#,# 256,
        # tune.grid_search(['adam', 'yogi', 'sgd']),# tune.grid_search(['adam', 'yogi']),# tune.grid_search(['adam', 'yogi', 'sgd']),
        'optimizer': 'yogi',
        # tune.loguniform(1e-6, 1),# tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),# tune.loguniform(1e-5, 1e-1),
        'lr': None,
        'main_loss': 'mse',  # tune.grid_search(['mse', 'bce-wl']),
        # tune.grid_search(['mix', 'gausk1', 'gausk3']),
        'mod_loss': tune.grid_search(['bce+dice', 'none']),
        'beta': 0.4,  # tune.grid_search([0.1, 0.2, 0.3, 0.4]),
        # tune.grid_search([0.0, 0.1, 0.2, 0.3]),# tune.uniform(0.0, 0.5),# tune.grid_search([0.0, 0.1, 0.2, 0.3,]),
        'corruption': 0.0,
        'noising': 0.0,  # tune.grid_search([0.0, 0.1]),
        'train_dec': 'yes',
        'alpha': 1,  # tune.grid_search([1, 9]),
        'scaler': tune.grid_search(['standard', 'normal-l1', 'normal-l2', 'none']),
        'use_emp_centroids': tune.grid_search(['yes', 'no']),
    }

    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[
            "ae_loss",
            "cl_recon",
            "accuracy",
            "cycle_accuracy",
            "delta_label",
            "training_iteration",
            "cos_sil_score",
            "eucl_sil_score",
            "data_calinski_harabasz",
            "feat_calinski_harabasz"]
    )

    def lambda_scheduler(x): return ReduceLROnPlateau(
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
        keep_checkpoints_num=2 if config['train_dec'] == 'no' else 3,
        checkpoint_at_end=True,
        # scheduler=scheduler,
        # search_alg=bayesopt,
        progress_reporter=reporter,
        name='euromds_ae_mod_loss_scaler',
        # resume=True,
    )

    # best reconstruction loss after weights initialization
    best_trial = result.get_best_trial("ae_loss", "min", "last")
    print("Best reconstruction loss config: {}".format(best_trial.config))
    print("Best reconstruction loss value: {}".format(
        best_trial.last_result["ae_loss"]))

    if config['train_dec'] == 'yes':
        # best accuracy w.r.t. hdp labels
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best accuracy w.r.t. hdp labels config: {}".format(best_trial.config))
        print("Best accuracy w.r.t. hdp labels value: {}".format(
            best_trial.last_result["accuracy"]))

        # best reconstruction loss after clustering stage
        best_trial = result.get_best_trial("cl_recon", "min", "last")
        print("Best reconstruction loss after clustering stage config: {}".format(
            best_trial.config))
        print("Best reconstruction loss after clustering stage value: {}".format(
            best_trial.last_result["cl_recon"]))

        # best euclidean silhouette after clustering stage
        best_trial = result.get_best_trial("eucl_sil_score", "max", "last")
        print("Best euclidean silhouette after clustering stage config: {}".format(
            best_trial.config))
        print("Best euclidean silhouette after clustering stage value: {}".format(
            best_trial.last_result["eucl_sil_score"]))

        # best cosine silhouette after clustering stage
        best_trial = result.get_best_trial("cos_sil_score", "max", "last")
        print("Best cosine silhouette after clustering stage config: {}".format(
            best_trial.config))
        print("Best cosine silhouette after clustering stage value: {}".format(
            best_trial.last_result["cos_sil_score"]))

        # best calinski harabasz score of data after clustering stage
        best_trial = result.get_best_trial(
            "data_calinski_harabasz", "max", "last")
        print("Best calinski harabasz score of data after clustering stage config: {}".format(
            best_trial.config))
        print("Best calinski harabasz score of data after clustering stage value: {}".format(
            best_trial.last_result["data_calinski_harabasz"]))

        # best calinski harabasz score of features after clustering stage
        best_trial = result.get_best_trial(
            "feat_calinski_harabasz", "max", "last")
        print("Best calinski harabasz score of features after clustering stage config: {}".format(
            best_trial.config))
        print("Best calinski harabasz score of features after clustering stage value: {}".format(
            best_trial.last_result["feat_calinski_harabasz"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
