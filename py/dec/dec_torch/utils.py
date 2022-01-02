from functools import partial
import numpy as np
import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment

from py.losses.torch import SobelLoss, GaussianBlurredLoss


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
    cuda: bool = False,
    ):
    main_loss = get_main_loss(main_loss)
    loss_dict = {
        'sobel': [partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, cuda)],
        'gausk1': [partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, cuda)],
        'gausk3': [partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, cuda)],
        'mix': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, cuda),
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, cuda),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, cuda),
        ],
        'mix-gk': [
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, cuda),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, cuda),
        ],
        'mix-s-gk1': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, cuda),
            partial(GaussianBlurredLoss, 1, beta, main_loss, unflatten, cuda),
        ],
        'mix-s-gk3': [
            partial(SobelLoss, beta, main_loss, main_loss!='mse', unflatten, True, cuda),
            partial(GaussianBlurredLoss, 3, beta, main_loss, unflatten, cuda),
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
