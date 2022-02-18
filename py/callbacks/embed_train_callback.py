import numpy as np
import torch
from torch.utils.data import DataLoader


def embed_train_callback(
    wrtr,
    additions,
    device,
    img_repr,
    config,
    name,
    epoch,
    dataset,
    autoencoder
):
    features = []
    images = []
    labels = []
    r_images = []
    dataloader = DataLoader(dataset, batch_size=config['ae_batch_size'], shuffle=True)
    for batch in dataloader:
        with torch.no_grad():
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
            batch = batch.to(device, non_blocking=True)
            features.append(autoencoder.encoder(batch).detach().cpu())
            r_images.append(autoencoder(batch).detach().cpu())
            images.append(batch.detach().cpu())
            labels.append(value.detach().cpu())
    images = torch.cat(images)
    r_images = torch.cat(r_images)
    if additions > 0:
        to_add = np.zeros(shape=(images.shape[0], additions))
        images = torch.cat((images, torch.Tensor(to_add)), 1)
        r_images = torch.cat((r_images, torch.Tensor(to_add)), 1)
    wrtr.add_embedding(
        torch.cat(features).numpy(), # Encodings per image
        metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
        label_img=images.reshape(img_repr).numpy(),  # Adding the original images to the plot
        global_step=2+epoch,
        tag=str(name+'{}'.format(config['alpha'])),
        )
    wrtr.add_embedding(
        torch.cat(features).numpy(), # Encodings per image
        metadata=torch.cat(labels).numpy(), # Adding the labels per image to the plot
        label_img=r_images.reshape(img_repr).numpy(),  # Adding the original images to the plot
        global_step=2+epoch,
        tag=str(name+'{}_r'.format(config['alpha'])),
        )
