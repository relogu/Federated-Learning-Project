from functools import partial
import numpy as np
from pyparsing import dictOf
import torch
from torch.optim import SGD, Adam
from torch_optimizer import Yogi 
from typing import Optional
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler, Normalizer

from py.losses.torch import (SobelLoss, GaussianBlurredLoss, ComboLoss,
                             DiceBCELoss, DiceLoss, FocalLoss, FocalTverskyLoss,
                             TverskyLoss, LovaszHingeLoss, IoULoss)


def get_linears(name, input_dim, f_dim):
    linears_dict = {
        'dec': [input_dim, 500, 500, 2000, f_dim],
        'google': [input_dim, 1000, 500, 250, f_dim],
        'curves': [input_dim, 400, 200, 100, 50, 25, 6],
    }
    return linears_dict[name]

def get_scaler(name: str):
    scaler_dict = {
        'standard': StandardScaler(),
        'normal-l1': Normalizer(norm='l1'),
        'normal-l2': Normalizer(norm='l2'),
    }
    return scaler_dict[name]

def get_ae_opt(name: str, dataset: str, lr: float = None):
    lr_dataset_dict = {
        'euromds': {
            'sgd': 5e-2,
            'adam': 3e-4,
            'yogi': 3.5e-2,
        },
        'mnist': {
            'sgd': 0.5,
            'adam': 3.5e-4,
            'yogi': 0.1,
        },
        'bmnist': {
            'sgd': 0.5,
            'adam': 3e-4,
            'yogi': 0.1,
        },
    }
    ae_opt_dict = {
        'sgd': partial(SGD,
                       lr=lr_dataset_dict[dataset]['sgd'] if lr is None else lr,
                       momentum=0.9),
        'adam': partial(Adam,
                        lr=lr_dataset_dict[dataset]['adam'] if lr is None else lr),
        'yogi': partial(Yogi,
                        lr=lr_dataset_dict[dataset]['yogi'] if lr is None else lr,
                        eps=1e-3,
                        initial_accumulator=1e-6,),
    }
    return ae_opt_dict[name]


def get_main_loss(name: str):
    loss_dict = {
        'mse': torch.nn.MSELoss,
        'ce': torch.nn.CrossEntropyLoss,
        'bce': torch.nn.BCELoss,
        'bce-wl': torch.nn.BCEWithLogitsLoss,
    }
    return loss_dict[name]


def get_mod_loss(
    name: str,
    main_loss: str,
    beta: float = 0.5,
    unflatten: bool = True,
    device: str = 'cpu',
    ):
    main_loss = get_main_loss(main_loss)
    loss_dict = {
        'sobel': [partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, device)],
        'gausk1': [partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, device)],
        'gausk3': [partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, device)],
        'mix': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, device),
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, device),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, device),
        ],
        'mix-gk': [
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, device),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, device),
        ],
        'mix-s-gk1': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, device),
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, device),
        ],
        'mix-s-gk3': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, device),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, device),
        ],
    }
    return loss_dict[name]


def get_mod_binary_loss(
    name: str,
    ):
    loss_dict = {
        'dice': [
            torch.nn.MSELoss,
            DiceLoss,
        ],
        'lovasz-hinge': [
            torch.nn.MSELoss,
            LovaszHingeLoss
        ],
        'iou': [
            torch.nn.MSELoss,
            IoULoss
        ],
        'combo': [
            torch.nn.MSELoss,
            ComboLoss
        ],
        'focal': [
            torch.nn.MSELoss,
            FocalLoss
        ],
        'tversky': [
            torch.nn.MSELoss,
            TverskyLoss
        ],
        'focal-tversky': [
            torch.nn.MSELoss,
            FocalTverskyLoss
        ],
        'bce+dice': [
            torch.nn.MSELoss,
            partial(DiceBCELoss, use_sigmoid=True)
        ],
    }
    return loss_dict[name]

def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
