from functools import partial
import os
from typing import Any, Callable, Optional, Iterable, List, Dict
import torch
import torch.nn.functional as F
from torch.nn import Module, ReLU, Dropout
from torch.nn.modules.loss import MSELoss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.dec.layers.torch import TruncatedGaussianNoise
from py.datasets.mnist import CachedMNIST
from py.dec.dec_torch.utils import get_ae_opt, get_main_loss, get_mod_loss

def train_ae(
    config: Dict,
    scheduler: Any = None,
    device: str = 'cpu',
) -> None:
    """
    Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.

    :param config: TODO
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param device: TODO
    :return: None
    """
    # get datasets
    ds_train = CachedMNIST(
        train=True, device=device
    )  # training dataset
    ds_val = CachedMNIST(
        train=False, device=device
    )  # evaluation dataset
    dataloader = DataLoader(
        ds_train,
        batch_size=config['batch_size'],
        shuffle=True,
    )
    validation_loader = DataLoader(
        ds_val,
        batch_size=config['batch_size'],
        shuffle=True,
    )
    
    # set up loss(es) used in training the SDAE
    if config['mod_loss'] is not None:
        loss_fn = get_mod_loss(
            name=config['mod_loss'],
            beta=config['beta'],
            main_loss=config['main_loss'],
            device=device,
            )
    else:
        loss_fn = [get_main_loss(config['main_loss'])]
    
    noising = None
    if config['noising'] is not None:
        noising = TruncatedGaussianNoise(
            shape=784,
            stddev=config['noising'],
            rate=1.0,
            device=device,
            )
        
    corruption = None
    if config['corruption'] is not None:
        corruption = config['corruption']
        
    loss_functions = [loss_fn_i() for loss_fn_i in loss_fn]
    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        config['linears'],
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
    validation_loss_value = -1
    for epoch in range(300):
        running_loss = 0.0
        epoch_steps = 0
        if scheduler is not None:
            scheduler.step(validation_loss_value)
            
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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((autoencoder.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


def main(num_samples=5, max_num_epochs=10, gpus_per_trial=1):
    
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda:0"

    # features space dimension
    z_dim = 10
    
    config = {
        'linears': [28 * 28, 500, 500, 2000, z_dim],
        'activation': ReLU(),
        'final_activation': ReLU(),
        'dropout': tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),#0.2,
        'epochs': 100,
        'batch_size': 256,
        'optimizer': 'adam',
        'lr': tune.loguniform(1e-4, 1e-1),#0.1,
        'main_loss': 'mse',
        'mod_loss': None,#'gausk1',
        'beta': 0.5,
        'corruption': tune.grid_search([0.0, 0.1, 0.2, 0.3]),#0.2,
        'noising': 0.0,#0.2,
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"]
        )
    
    lambda_scheduler = lambda x: ReduceLROnPlateau(
        x,
        mode='min',
        factor=0.5,
        patience=20,
    )
                
    result = tune.run(
        partial(train_ae, scheduler=lambda_scheduler, device=device),
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

    # best_trained_model = StackedDenoisingAutoEncoder(
    #     dimensions=best_trial.config["linear"],
    #     activation=best_trial.config["activation"],
    #     final_activation=best_trial.config["final_activation"],
    #     dropout=best_trial.config["dropout"],
    #     is_tied=True,
    #     )
    
    # if gpus_per_trial > 1:
    #     best_trained_model = torch.nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
    
