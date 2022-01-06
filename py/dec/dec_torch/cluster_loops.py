import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
import torch
from torch.nn import Module, KLDivLoss
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.sampler import Sampler
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
import copy

from .utils import target_distribution, cluster_accuracy
from py.util import compute_centroid_np


def train(
    dataset: torch.utils.data.Dataset,
    model: Module,
    epochs: int,
    batch_size: int,
    optimizer: Optimizer,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    # cuda: bool = True,
    device: str = 'cpu',
    sampler: Optional[Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, Module], None]] = None,
) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param optimizer: instance of optimizer to use
    :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether to use CUDA, defaults to True
    :param device: TODO
    :param sampler: optional sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 10
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param update_callback: optional function of accuracy and loss to update, default None
    :param epoch_callback: optional function of epoch and model, default None
    :return: None
    """
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True,
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )
    '''
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(model.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    '''
    predicted_previous, accuracy = assign_cluster_centers(
        dataset=dataset,
        model=model,
        batch_size=batch_size,
        collate_fn=collate_fn,
        # cuda=cuda,
        device=device,
        sampler=sampler,
        silent=silent
    )
    loss_function = KLDivLoss(size_average=False)
    delta_label = None
    for epoch in range(epochs):
        # predicted_previous, accuracy = assign_cluster_centers(
        #     dataset=dataset,
        #     model=model,
        #     batch_size=batch_size,
        #     collate_fn=collate_fn,
        #     # cuda=cuda,
        #     device=device,
        #     sampler=sampler,
        #     silent=silent
        # )
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
            },
            disable=silent,
        )
        # old_model = copy.deepcopy(model)
        model.train()
        for index, batch in enumerate(data_iterator):
            # if index % 140:
            #     old_model = copy.deepcopy(model)
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            # if cuda:
            #     batch = batch.cuda(non_blocking=True)
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            soft_labels = output
            # soft_labels = old_model(batch)
            target = target_distribution(soft_labels).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            features.append(model.encoder(batch).detach().cpu())
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(
                        epoch,
                        optimizer.param_groups[0]["lr"],
                        accuracy,
                        loss_value,
                        1.0 if delta_label is None else delta_label
                        )
        predicted, actual = predict(
            dataset,
            model,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            # cuda=cuda,
            device=device,
        )
        delta_label = (
            float((predicted != predicted_previous).float().sum().item())
            / predicted_previous.shape[0]
        )
        if stopping_delta is not None and delta_label < stopping_delta:
            print(
                'Early stopping as label delta "%1.5f" less than "%1.5f".'
                % (delta_label, stopping_delta)
            )
            break
        predicted_previous = predicted
        _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        data_iterator.set_postfix(
            epo=epoch,
            acc="%.4f" % (accuracy or 0.0),
            lss="%.8f" % 0.0,
            dlb="%.4f" % (delta_label or 0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model)


def predict(
    dataset: torch.utils.data.Dataset,
    model: Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    # cuda: bool = True,
    device: str = 'cpu',
    silent: bool = False,
    return_actual: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param device: TODO
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        # if cuda:
        #     batch = batch.cuda(non_blocking=True)
        batch = batch.to(device, non_blocking=True)
        features.append(
            model(batch).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]
    

def assign_cluster_centers(
    dataset: torch.utils.data.Dataset,
    model: Module,
    batch_size: int,
    collate_fn=default_collate,
    # cuda: bool = True,
    device: str = 'cpu',
    sampler: Optional[Sampler] = None,
    silent: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TODO

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param device: TODO
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    """
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        # postfix={
        #     "epo": -1,
        #     "acc": "%.4f" % 0.0,
        #     "lss": "%.8f" % 0.0,
        #     "dlb": "%.4f" % -1,
        # },
        disable=True,
    )
    # scaler = StandardScaler()
    scaler = Normalizer(norm='l1')
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    #model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        # if cuda:
        #     batch = batch.cuda(non_blocking=True)
        batch = batch.to(device, non_blocking=True)
        features.append(model.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    
    predicted = kmeans.fit_predict(scaler.fit_transform(torch.cat(features).numpy()))
    # predicted = kmeans.fit_predict(normalize(torch.cat(features).numpy()))
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    
    emp_centroids = []
    for i in np.unique(predicted):
        idx = (predicted == i)
        emp_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))
    
    # true_centroids = []
    # for i in np.unique(actual.cpu().numpy()):
    #     idx = (actual.cpu().numpy() == i)
    #     true_centroids.append(compute_centroid_np(torch.cat(features).numpy()[idx, :]))
    
    cluster_centers = torch.tensor(
        np.array(emp_centroids),#kmeans.cluster_centers_,np.array(true_centroids),#
        dtype=torch.float,
        requires_grad=False,#True
    )
    # if cuda:
    #     cluster_centers = cluster_centers.cuda(non_blocking=True)
    cluster_centers = cluster_centers.to(device, non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    
    return predicted_previous, accuracy