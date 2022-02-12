import os
import pathlib
import click
from functools import partial
import numpy as np

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

from py.dec.dec_torch.dec import DEC
from py.dec.dec_torch.cluster_loops import train, predict
from py.dec.dec_torch.sdae import StackedDenoisingAutoEncoder
from py.dec.layers.torch import TruncatedGaussianNoise
import py.dec.dec_torch.ae_loops as ae
from py.dec.dec_torch.utils import cluster_accuracy, get_main_loss, get_mod_binary_loss, get_ae_opt
from py.datasets.euromds import CachedEUROMDS
from py.util import get_square_image_repr


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
    type=str,#click.Choice(['mse+dice', 'combo', 'bce+dice']),
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
@click.option(
    '--ae-opt',
    type=click.Choice(['sgd', 'adam', 'yogi']),
    default='sgd',
    help='Optimizer for AE training (default sgd)'
)
@click.option(
    "--lr",
    help="value for learning rate of AE opt (default 0.01).",
    type=float,
    default=0.01,
)
@click.option(
    '--path-to-data',
    type=str,
    default=None,
    help='Path to data (default None)'
)
def main(cuda, gpu_id, batch_size, pretrain_epochs, finetune_epochs, testing_mode, out_folder,
         glw_pretraining, is_tied, ae_main_loss, ae_mod_loss, alpha, input_do, hidden_do, beta,
         gaus_noise, ae_opt, lr, path_to_data):
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
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:{}".format(gpu_id)
    
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
        ae_mod_loss_fn = get_mod_binary_loss(
            name=ae_mod_loss,
            # beta=beta,
            # main_loss=ae_main_loss,
            # device=device,
            )
    else:
        ae_mod_loss_fn = [get_main_loss(ae_main_loss)]
        
    # set up optimizer used in training the SDAE
    ae_opt_fn = get_ae_opt(ae_opt, lr)
    
    # get datasets
    path_to_data = pathlib.Path('/home/relogu/Desktop/OneDrive/UNIBO/Magistrale/Federated Learning Project/data/euromds') if path_to_data is None else pathlib.Path(path_to_data)
    ds_train = CachedEUROMDS(
        exclude_cols=['UTX', 'CSF3R', 'SETBP1', 'PPM1D'],
        groups=['Genetics', 'CNA'],
        path_to_data=path_to_data,
        fill_nans=2044,
        get_hdp=True,
        get_outcomes=True,
        get_ids=True,
        verbose=True,
        device=device,
    )  # training dataset
    ds_val = ds_train
    # img_repr = get_image_repr(ds_train.n_features)
    img_repr = get_square_image_repr(ds_train.n_features)
    hdp_labels = np.array([hdp_row.argmax() for hdp_row in ds_train.hdp])
    
    print("Square image representation for {} features is (x,y,add): {}".format(ds_train.n_features, img_repr))
    additions = img_repr[2]
    img_repr = (-1, 1, img_repr[0], img_repr[1])

    # set noising to data        
    if gaus_noise:
        noising = TruncatedGaussianNoise(
            shape=ds_train.n_features,
            stddev=input_do,
            rate=1.0,
            device=device,
            )
    else:
        noising = None
        
    # features space dimension
    z_dim = ds_train.hdp.shape[1] # 10
    # AE layers' dimension
    # linears = [ds_train.n_features, 1000, 500, 250, z_dim]
    linears = [ds_train.n_features, 500, 500, 2000, z_dim]
    
    # set up SDAE
    autoencoder = StackedDenoisingAutoEncoder(
        linears,
        # activation=torch.nn.ReLU(),
        activation=torch.nn.Sigmoid(),
        # final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
        final_activation=torch.nn.Sigmoid(),
        dropout=hidden_do,
        is_tied=is_tied,
    )
    
    # if torch.cuda.device_count() > 1:
    #     autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.to(device)
        
    if (path_to_out/'pretrain_ae').exists():
        print('Skipping pretraining since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
    else:
        print("Pretraining stage.")
        if glw_pretraining:
            # greedy layer-wise pretraining
            lambda_ae_opt = lambda model: ae_opt_fn(params=model.parameters())
            lambda_scheduler = lambda x: StepLR(x, 100, gamma=0.1)
            ae.pretrain(
                ds_train,
                autoencoder,
                loss_fn=[ae_main_loss_fn],
                final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
                device=device,
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
            lambda_ae_opt = lambda model: ae_opt_fn(params=model.parameters())
            lambda_scheduler = lambda x: ReduceLROnPlateau(
                x,
                mode='min',
                factor=0.5,
                patience=20,)# None
            ae.train(
                ds_train,
                autoencoder,
                loss_fn=ae_mod_loss_fn,
                device=device,
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
            dataloader = DataLoader(ds_val, batch_size=1024, shuffle=False)
            for i, batch in enumerate(dataloader):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    batch, value = batch  # if we have a prediction label, separate it to actual
                batch = batch.to(device, non_blocking=True)
                features.append(autoencoder.encoder(batch).detach().cpu())
                r_images.append(autoencoder(batch).detach().cpu())
                labels.append(value.detach().cpu())
                images.append(batch.detach().cpu())
            images = torch.cat(images)
            r_images = torch.cat(r_images)
            if additions > 0:
                to_add = np.zeros(shape=(images.shape[0], additions))
                images = torch.cat((images, torch.Tensor(to_add)), 1)
                r_images = torch.cat((r_images, torch.Tensor(to_add)), 1)
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=images.reshape(img_repr).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='pretraining',
                )
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=r_images.reshape(img_repr).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='pretraining_r',
                )
            np.savez(path_to_out/'pretrain_ae_features', torch.cat(features).numpy())
    if (path_to_out/'finetune_ae').exists():
        print('Skipping finetuning since weights already exist.')
        autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    else:
        print("Training stage.")
        # finetuning
        autoencoder = StackedDenoisingAutoEncoder(
            linears,
            final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
            dropout=hidden_do,
            is_tied=is_tied,
        )
        autoencoder.load_state_dict(torch.load(path_to_out/'pretrain_ae'))
        if cuda:
            autoencoder.cuda()
        if glw_pretraining:
            ae_opt = ae_opt_fn(params=autoencoder.parameters())
            scheduler = StepLR(ae_opt, 100, gamma=0.1)
        else:
            ae_opt = ae_opt_fn(params=autoencoder.parameters())
            scheduler = ReduceLROnPlateau(
                ae_opt,
                mode='min',
                factor=0.5,
                patience=20,)
        ae.train(
            ds_train,
            autoencoder,
            loss_fn=ae_mod_loss_fn,
            device=device,
            validation=ds_val,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_opt,
            scheduler=scheduler,
            # corruption=input_do,
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
                batch = batch.to(device, non_blocking=True)
                features.append(autoencoder.encoder(batch).detach().cpu())
                r_images.append(autoencoder(batch).detach().cpu())
                images.append(batch.detach().cpu())
                labels.append(value.detach().cpu())
            np.savez(path_to_out/'finetune_ae_features', torch.cat(features).numpy())
            images = torch.cat(images)
            r_images = torch.cat(r_images)
            if additions > 0:
                to_add = np.zeros(shape=(images.shape[0], additions))
                images = torch.cat((images, torch.Tensor(to_add)), 1)
                r_images = torch.cat((r_images, torch.Tensor(to_add)), 1)
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=images.reshape(img_repr).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='finetuning',
                )
            writer.add_embedding(
                torch.cat(features).numpy(), # Encodings per image
                metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
                label_img=r_images.reshape(img_repr).numpy(),  # Adding the original images to the plot
                global_step=0,
                tag='finetuning_r',
                )
            np.savez(path_to_out/'finetune_ae_features', torch.cat(features).numpy())
    print("DEC stage.")
    # autoencoder = StackedDenoisingAutoEncoder(
    #     linears,
    #     final_activation=torch.nn.Sigmoid() if ae_main_loss == 'bce' else torch.nn.ReLU(),
    #     dropout=hidden_do,
    #     is_tied=is_tied,
    # )
    autoencoder.load_state_dict(torch.load(path_to_out/'finetune_ae'))
    
    autoencoder = autoencoder.to(device)
    # callback function to call during training, uses writer from the scope
    def training_callback1(alpha, epoch, lr, accuracy, loss, delta_label):
        writer.add_scalars(
            "data/clustering_alpha{}".format(alpha),
            {"lr": lr, "accuracy": accuracy, "loss": loss, "delta_label": delta_label,},
            epoch,
        )
    # callback function to call at each epoch end
    def epoch_callback1(epoch, model):
        features = []
        images = []
        labels = []
        r_images = []
        dataloader = DataLoader(ds_train, batch_size=1024, shuffle=True)
        for batch in dataloader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
            batch = batch.to(device, non_blocking=True)
            features.append(model.encoder(batch).detach().cpu())
            r_images.append(autoencoder(batch).detach().cpu())
            images.append(batch.detach().cpu())
            labels.append(value.detach().cpu())
        images = torch.cat(images)
        r_images = torch.cat(r_images)
        if additions > 0:
            to_add = np.zeros(shape=(images.shape[0], additions))
            images = torch.cat((images, torch.Tensor(to_add)), 1)
            r_images = torch.cat((r_images, torch.Tensor(to_add)), 1)
        writer.add_embedding(
            torch.cat(features).numpy(), # Encodings per image
            metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
            label_img=images.reshape(img_repr).numpy(),  # Adding the original images to the plot
            global_step=2+epoch,
            tag='clustering_alpha{}'.format(alpha),
            )
        writer.add_embedding(
            torch.cat(features).numpy(), # Encodings per image
            metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
            label_img=r_images.reshape(img_repr).numpy(),  # Adding the original images to the plot
            global_step=2+epoch,
            tag='clustering_alpha{}_r'.format(alpha),
            )
    model = DEC(cluster_number=10,
                hidden_dimension=z_dim,
                encoder=autoencoder.encoder,
                alpha=alpha)
    model = model.to(device)
    if glw_pretraining:
        dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        dec_optimizer = Adam(params=model.parameters(), lr=1e-3)
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # dec_optimizer = Adam(params=model.parameters(), lr=1e-4)
    train(
        dataset=ds_train,
        model=model,
        epochs=20,# 100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        update_callback=partial(training_callback1, alpha),
        epoch_callback=epoch_callback1,
        device=device,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, device=device
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
        batch = batch.to(device, non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
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
