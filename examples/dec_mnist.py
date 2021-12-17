import os
import pathlib
import click
from functools import partial
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from tensorboardX import SummaryWriter
import uuid

from py.losses.torch import SobelLoss, GaussianBlurredLoss

from py.dec.dec_torch.dec import DEC
from py.dec.dec_torch.cluster_loops import train, predict
from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
import py.dec.dec_torch.ae_loops as ae
from py.dec.dec_torch.utils import cluster_accuracy, get_main_loss, get_mod_loss
from py.datasets.mnist import CachedMNIST


@click.command()
@click.option(
    "--cuda",
    help="whether to use CUDA (default False).",
    type=bool,
    default=False
)
@click.option(
    "--gpu-id",
    help="id of the GPU to use",
    type=int,
    default=0,
)
@click.option(
    "--batch-size",
    help="training batch size (default 256).",
    type=int,
    default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=500,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
@click.option(
    "--out-folder",
    help="folder for dumping results",
    type=str,
    default=False,
)
# customized arguments
@click.option(
    "--is-tied",
    help="whether to use tied weights for the SDAE (the training procedure changes accordingly, default False)",
    type=bool,
    default=False
)
@click.option(
    '--ae-main-loss',
    type=click.Choice(['mse', 'bce']),
    default='mse',
    help='Main loss function for autoencoder training'
)
@click.option(
    '--ae-mod-loss',
    type=click.Choice(['sobel', 'gausk1', 'gausk3']),
    default=None,
    help='Modified loss function for autoencoder training'
)
@click.option(
    "--alpha",
    help="value for parameter alpha (d-o-f for auxiliary distr., default 1).",
    type=int,
    default=1,
)
def main(cuda, gpu_id, batch_size, pretrain_epochs, finetune_epochs, testing_mode, out_folder,
         is_tied, ae_main_loss, ae_mod_loss, alpha):
    writer = SummaryWriter()  # create the TensorBoard object
    # defining output folder
    if out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(out_folder)
    os.makedirs(path_to_out, exist_ok=True)
    print('Output folder {}'.format(path_to_out))
    
    #gpus = tf.config.list_physical_devices('GPU')
    #tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
    if cuda:
        torch.cuda.set_device(gpu_id)
    # callback function to call during training, uses writer from the scope
    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    ds_train = CachedMNIST(
        train=True, cuda=cuda, testing_mode=testing_mode
    )  # training dataset
    ds_val = CachedMNIST(
        train=False, cuda=cuda, testing_mode=testing_mode
    )  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=torch.nn.ReLU(),
        dropout=0.2,
        is_tied=is_tied,
    )
    if cuda:
        autoencoder.cuda()
    
    ae_main_loss_fn = get_main_loss(ae_main_loss)
    if ae_mod_loss is not None:
        ae_mod_loss_fn = get_mod_loss(
            name=ae_mod_loss,
            main_loss=ae_main_loss,
            cuda=cuda)
    else:
        ae_mod_loss_fn = get_main_loss(ae_main_loss)
        
    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        # lambda_ae_opt = lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9)
        # lambda_scheduler = lambda x: StepLR(x, 100, gamma=0.1)
        lambda_ae_opt = lambda model: Adam(model.parameters(), lr=1e-4)
        lambda_scheduler = lambda x: None
        if is_tied:
            ae.train(
                ds_train,
                autoencoder,
                loss_fn=ae_mod_loss_fn,
                cuda=cuda,
                validation=ds_val,
                epochs=pretrain_epochs,
                batch_size=batch_size,
                optimizer=lambda_ae_opt(autoencoder),
                scheduler=lambda_scheduler(lambda_ae_opt(autoencoder)),
                corruption=0.4,
                update_callback=training_callback,
            )
        else:
            ae.pretrain(
                ds_train,
                autoencoder,
                loss_fn=ae_main_loss_fn,
                cuda=cuda,
                validation=ds_val,
                epochs=pretrain_epochs,
                batch_size=batch_size,
                optimizer=lambda_ae_opt,
                scheduler=lambda_scheduler,
                corruption=0.4,
            )
        torch.save(autoencoder.state_dict(), path_to_out/'pretrain_ae')
    print('Saving features after pretraining.')
    autoencoder.eval()
    if not testing_mode:
        features = []
        dataloader = DataLoader(ds_train, batch_size=1024, shuffle=False)
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
            if cuda:
                batch = batch.cuda(non_blocking=True)
            features.append(autoencoder.encoder(batch).detach().cpu())
        np.savez(path_to_out/'pretrain_ae_features', torch.cat(features).numpy())
    if (path_to_out/'finetune_ae').exists():
        print('Skipping finetuning since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    else:
        print("Training stage.")
        autoencoder = StackedDenoisingAutoEncoder(
            [28 * 28, 500, 500, 2000, 10],
            final_activation=torch.nn.ReLU(),
            dropout=0.2,
            is_tied=is_tied,
        )
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
        if cuda:
            autoencoder.cuda()
        # ae_opt = SGD(autoencoder.parameters(), lr=0.1, momentum=0.9)
        # scheduler = StepLR(ae_opt, 100, gamma=0.1)
        ae_opt = Adam(autoencoder.parameters(), lr=1e-4)
        scheduler = None
        ae.train(
            ds_train,
            autoencoder,
            loss_fn=ae_mod_loss_fn,
            cuda=cuda,
            validation=ds_val,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_opt,
            scheduler=scheduler,
            corruption=0.2,
            update_callback=training_callback,
        )
        torch.save(autoencoder.state_dict(), path_to_out/'finetune_ae')
    print('Saving features after finetuning.')
    autoencoder.eval()
    if not testing_mode:
        features = []
        dataloader = DataLoader(ds_train, batch_size=1024, shuffle=False)
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
            if cuda:
                batch = batch.cuda(non_blocking=True)
            features.append(autoencoder.encoder(batch).detach().cpu())
        np.savez(path_to_out/'finetune_ae_features', torch.cat(features).numpy())
    print("DEC stage.")
    model = DEC(cluster_number=10,
                hidden_dimension=10,
                encoder=autoencoder.encoder,
                alpha=alpha)
    if cuda:
        model.cuda()
    # dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    dec_optimizer = Adam(params=model.parameters(), lr=1e-4)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print('Saving features, predictions and true labels after clustering.')
    autoencoder.eval()
    features = []
    actual = []
    dataloader = DataLoader(ds_train, batch_size=1024, shuffle=False)
    for batch in dataloader:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
    np.savez(path_to_out/'final_ae_features', torch.cat(features).numpy())
    print("Final DEC accuracy: %s" % accuracy)
    torch.save(autoencoder.state_dict(), path_to_out/'final_ae')
    torch.save(model.state_dict(), path_to_out/'clustering_model')
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        np.savez(path_to_out/'final_assignments', predicted_reassigned)
        np.savez(path_to_out/'actual_labels', torch.cat(actual).detach().cpu().numpy())
        # confusion = confusion_matrix(actual, predicted_reassigned)
        # normalised_confusion = (
        #     confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        # )
        # confusion_id = uuid.uuid4().hex
        # sns.heatmap(normalised_confusion).get_figure().savefig(
        #     "confusion_%s.png" % confusion_id
        # )
        # print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    main()
