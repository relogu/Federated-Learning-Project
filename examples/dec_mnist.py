import os
import pathlib
import click
from functools import partial
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch_optimizer import Yogi 
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from tensorboardX import SummaryWriter
import uuid

from py.losses.torch import SobelLoss, GaussianBlurredLoss

from py.dec.dec_torch.dec import DEC
from py.dec.dec_torch.cluster_loops import train, predict
from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.dec.layers.torch import TruncatedGaussianNoise
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
    help="folder for dumping results (default None)",
    type=str,
    default=False,
)
# customized arguments
@click.option(
    "--glw-pretraining",
    help="whether to use greedy layer-wise pretraining(default True)",
    type=bool,
    default=True
)
@click.option(
    "--is-tied",
    help="whether to use tied weights for the SDAE (default False)",
    type=bool,
    default=False
)
@click.option(
    '--ae-main-loss',
    type=click.Choice(['mse', 'bce', 'bce-wl']),
    default='mse',
    help='Main loss function for autoencoder training (default mse)'
)
@click.option(
    '--ae-mod-loss',
    type=click.Choice(['sobel', 'gausk1', 'gausk3', 'mix', 'mix-gk', 'mix-s-gk1', 'mix-s-gk3']),
    default=None,
    help='Modified loss function for autoencoder training (default None)'
)
@click.option(
    "--alpha",
    help="value for parameter alpha (d-o-f for auxiliary distr., default 1).",
    type=int,
    default=1,
)
@click.option(
    "--input-do",
    help="value for dropout of input in pretraining and finetuning (default 0.2).",
    type=float,
    default=0.2,
)
@click.option(
    "--hidden-do",
    help="value for dropout of hidden linear layers of AE (default 0.5).",
    type=float,
    default=0.5,
)
@click.option(
    "--beta",
    help="value for scaling multiple losses (default 0.5).",
    type=float,
    default=0.5,
)
@click.option(
    "--gaus-noise",
    help="whether to apply gaussian noise in input at pretraining stage (default False)",
    type=bool,
    default=False
)
def main(cuda, gpu_id, batch_size, pretrain_epochs, finetune_epochs, testing_mode, out_folder,
         glw_pretraining, is_tied, ae_main_loss, ae_mod_loss, alpha, input_do, hidden_do, beta,
         gaus_noise):
    # defining output folder
    if out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(out_folder)
    os.makedirs(path_to_out, exist_ok=True)
    print('Output folder {}'.format(path_to_out))
    writer = SummaryWriter(
        logdir=str('runs/'+str(path_to_out)),
        flush_secs=5)  # create the TensorBoard object
    
    if cuda:
        torch.cuda.set_device(gpu_id)
    torch.autograd.set_detect_anomaly(True)
    # callback function to call during training, uses writer from the scope
    def training_callback(name, epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder_{}".format(name),
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )
    
    # set up loss(es) used in training the SDAE
    ae_main_loss_fn = get_main_loss(ae_main_loss)
    if ae_mod_loss is not None:
        ae_mod_loss_fn = get_mod_loss(
            name=ae_mod_loss,
            beta=beta,
            main_loss=ae_main_loss,
            cuda=cuda)
    else:
        ae_mod_loss_fn = [get_main_loss(ae_main_loss)]
        
    # set noising to data        
    if gaus_noise:
        noising = TruncatedGaussianNoise(
            shape=784,
            stddev=input_do,
            rate=1.0,
            cuda=cuda)
    else:
        noising = None
        
    # features space dimension
    z_dim = 10
    # learning rate for Adam
    adam_lr = 1
    # AE layers' dimension
    #linears = [28 * 28, 1000, 500, 250, z_dim]
    linears = [28 * 28, 500, 500, 2000, z_dim]
    
    # get datasets
    ds_train = CachedMNIST(
        train=True, cuda=cuda, testing_mode=testing_mode
    )  # training dataset
    ds_val = CachedMNIST(
        train=False, cuda=cuda, testing_mode=testing_mode
    )  # evaluation dataset
    
    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        linears,
        #activation=torch.nn.ReLU(),
        activation=torch.nn.Sigmoid(),
        #final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
        final_activation=torch.nn.Sigmoid(),
        dropout=hidden_do,
        is_tied=is_tied,
    )
    if cuda:
        autoencoder.cuda()
        
    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        if glw_pretraining:
            # greedy layer-wise pretraining
            lambda_ae_opt = lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9)
            lambda_scheduler = lambda x: StepLR(x, 100, gamma=0.1)
            ae.pretrain(
                ds_train,
                autoencoder,
                loss_fn=[ae_main_loss_fn],
                final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
                cuda=cuda,
                validation=ds_val,
                epochs=pretrain_epochs,
                batch_size=batch_size,
                optimizer=lambda_ae_opt,
                scheduler=lambda_scheduler,
                corruption=input_do if noising is None else None,
                noising=noising,
                update_callback=training_callback,
            )
        else:
            # pretraining with standard methods (may be apply some kind of noise to data that is not d/o)
            lambda_ae_opt = lambda model: Yogi(
                model.parameters(),
                lr=adam_lr,
                betas=(0.9, 0.999),
                eps=1e-3,
                initial_accumulator=1e-6,
                weight_decay=0)  # Adam(model.parameters(), lr=adam_lr)
            lambda_scheduler = lambda x: ReduceLROnPlateau(
                x,
                mode='min',
                factor=0.5,
                patience=20,)#None
            ae.train(
                ds_train,
                autoencoder,
                loss_fn=ae_mod_loss_fn,#[ae_main_loss_fn],#
                cuda=cuda,
                validation=ds_val,
                epochs=pretrain_epochs,
                batch_size=batch_size,
                optimizer=lambda_ae_opt(autoencoder),
                scheduler=lambda_scheduler(lambda_ae_opt(autoencoder)),
                corruption=input_do if noising is None else None,
                noising=noising,
                update_callback=partial(training_callback, 'pretraining'),
            )
        torch.save(autoencoder.state_dict(), path_to_out/'pretrain_ae')
        print('Saving features after pretraining.')
        autoencoder.eval()
        if not testing_mode:
            features = []
            labels = []
            images = []
            r_images = []
            dataloader = DataLoader(ds_train, batch_size=1024, shuffle=True)
            for i, batch in enumerate(dataloader):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    batch, value = batch  # if we have a prediction label, separate it to actual
                if cuda:
                    batch = batch.cuda(non_blocking=True)
                features.append(autoencoder.encoder(batch).detach().cpu())
                r_images.append(autoencoder(batch).detach().cpu())
                labels.append(value.detach().cpu())
                images.append(batch.detach().cpu())
                if i > 9:
                    break
            np.savez(path_to_out/'pretrain_ae_features', torch.cat(features).numpy())
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=torch.cat(images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='pretraining',
                )
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=torch.cat(r_images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='pretraining_r',
                )
    if (path_to_out/'finetune_ae').exists():
        print('Skipping finetuning since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    else:
        print("Training stage.")
        # finetuning
        autoencoder = StackedDenoisingAutoEncoder(
            linears,
            final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
            #dropout=hidden_do,
            is_tied=is_tied,
        )
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
        if cuda:
            autoencoder.cuda()
        if glw_pretraining:
            ae_opt = SGD(autoencoder.parameters(), lr=0.1, momentum=0.9)
            scheduler = StepLR(ae_opt, 100, gamma=0.1)
        else:
            ae_opt = Yogi(
                autoencoder.parameters(),
                lr=adam_lr,
                betas=(0.9, 0.999),
                eps=1e-3,
                initial_accumulator=1e-6,
                weight_decay=0)  # Adam(model.parameters(), lr=adam_lr)
            scheduler = ReduceLROnPlateau(
                ae_opt,
                mode='min',
                factor=0.5,
                patience=20,)#None
        ae.train(
            ds_train,
            autoencoder,
            loss_fn=ae_mod_loss_fn,#[ae_main_loss_fn],
            cuda=cuda,
            validation=ds_val,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_opt,
            scheduler=scheduler,
            #corruption=input_do,
            noising=None,
            update_callback=partial(training_callback, 'finetuning'),
        )
        torch.save(autoencoder.state_dict(), path_to_out/'finetune_ae')
        print('Saving features after finetuning.')
        autoencoder.eval()
        if not testing_mode:
            features = []
            labels = []
            images = []
            r_images = []
            dataloader = DataLoader(ds_train, batch_size=1024, shuffle=True)
            for i, batch in enumerate(dataloader):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    batch, value = batch  # if we have a prediction label, separate it to actual
                if cuda:
                    batch = batch.cuda(non_blocking=True)
                features.append(autoencoder.encoder(batch).detach().cpu())
                r_images.append(autoencoder(batch).detach().cpu())
                labels.append(value.detach().cpu())
                images.append(batch.detach().cpu())
                if i > 9:
                    break
            np.savez(path_to_out/'finetune_ae_features', torch.cat(features).numpy())
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=torch.cat(images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
                global_step=1,
                tag='finetuning',
                )
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=torch.cat(r_images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
                global_step=1,
                tag='finetuning_r',
                )
    print("DEC stage.")
    # autoencoder = StackedDenoisingAutoEncoder(
    #     linears,
    #     final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
    #     dropout=hidden_do,
    #     is_tied=is_tied,
    # )
    autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    if cuda:
        autoencoder.cuda()
    # callback function to call during training, uses writer from the scope
    def training_callback1(alpha, epoch, lr, accuracy, loss, delta_label):
        writer.add_scalars(
            "data/clustering_alpha{}".format(alpha),
            {"lr": lr, "accuracy": accuracy, "loss": loss, "delta_label": delta_label,},
            epoch,
        )
    # callback function to call at eah epoch end
    def epoch_callback1(epoch, model):
        features = []
        labels = []
        images = []
        r_images = []
        dataloader = DataLoader(ds_train, batch_size=1024, shuffle=True)
        for i, batch in enumerate(dataloader):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
            if cuda:
                batch = batch.cuda(non_blocking=True)
            features.append(model.encoder(batch).detach().cpu())
            r_images.append(autoencoder(batch).detach().cpu())
            labels.append(value.detach().cpu())
            images.append(batch.detach().cpu())
            if i > 9:
                break
        writer.add_embedding(
            torch.cat(features).numpy(), # Encodings per image
            metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
            label_img=torch.cat(images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
            global_step=2+epoch,
            tag='clustering_alpha{}'.format(alpha),
            )
        writer.add_embedding(
            torch.cat(features).numpy(), # Encodings per image
            metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
            label_img=torch.cat(r_images).reshape((-1, 1, 28, 28)).numpy(),  # Adding the original images to the plot
            global_step=2+epoch,
            tag='clustering_alpha{}_r'.format(alpha),
            )
    model = DEC(cluster_number=10,
                hidden_dimension=z_dim,
                encoder=autoencoder.encoder,
                alpha=alpha)
    if cuda:
        model.cuda()
    if glw_pretraining:
        dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        dec_optimizer = Adam(params=model.parameters(), lr=1e-3)
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    #dec_optimizer = Adam(params=model.parameters(), lr=1e-4)
    train(
        dataset=ds_train,
        model=model,
        epochs=20,#100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        update_callback=partial(training_callback1, alpha),
        epoch_callback=epoch_callback1,
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
    images = []
    dataloader = DataLoader(ds_train, batch_size=1024, shuffle=False)
    for batch in dataloader:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
        images.append(batch.detach().cpu())
    np.savez(path_to_out/'final_ae_features_alpha{}'.format(alpha), torch.cat(features).numpy())
    print("Final DEC accuracy: %s" % accuracy)
    torch.save(autoencoder.state_dict(), path_to_out/'final_ae_alpha{}'.format(alpha))
    torch.save(model.state_dict(), path_to_out/'clustering_model_alpha{}'.format(alpha))
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        np.savez(path_to_out/'final_assignments_alpha{}'.format(alpha), predicted_reassigned)
        np.savez(path_to_out/'actual_labels_alpha{}'.format(alpha), torch.cat(actual).detach().cpu().numpy())
        writer.close()


if __name__ == "__main__":
    main()
