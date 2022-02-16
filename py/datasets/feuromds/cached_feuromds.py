from typing import List, Union
from pathlib import Path
import numpy as np
from ray import client
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from py.dataset_util import (get_euromds_dataset, get_euromds_ids,
                             get_outcome_euromds_dataset, fillcolumn_prob)
from py.util import return_not_binary_indices, get_f_indices


class CachedfEUROMDS(Dataset):
    
    def __init__(self,
                 client_id: int,
                 exclude_cols: List[str] = None,
                 groups: List[str] = None,
                 path_to_data: Union[Path, str] = None,
                 fill_nans: int = 0,
                 get_hdp: bool = True,
                 get_outcomes: bool = False,
                 get_ids: bool = False,
                 verbose: bool = False,
                 n_clients: int = 1,
                 balance: float = 1.0,
                 seed: int = 51550,
                 device: str = 'cpu'):
        self.ds = np.array(get_euromds_dataset(
            path_to_data=path_to_data,
            groups=groups,
            exclude_cols=exclude_cols,
            accept_nan=fill_nans,
            fill_fn=fillcolumn_prob,
            verbose=verbose)).astype(np.float32)
        self.n_features = self.ds.shape[1]

        self.hdp = np.array(get_euromds_dataset(
            path_to_data=path_to_data, groups=['HDP'])) if get_hdp else None

        self.y = self.hdp.argmax(1) if self.hdp is not None else None

        self.outcomes = np.array(get_outcome_euromds_dataset(path_to_data=path_to_data)[
                                 ['outcome_3', 'outcome_2']]) if get_outcomes else None

        self.ids = np.array(get_euromds_ids(
            path_to_data=path_to_data)) if get_ids else None
        self.indices = get_f_indices(self.ds.shape[0], balance, n_clients, client_id, seed, verbose)
        
        
        self._cache = dict()
        self.device = device
    
    
    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            idx = self.indices[index]
            self._cache[index] = [torch.tensor(list(self.ds[idx])), torch.tensor(self.y[idx])]
            self._cache[index][0] = self._cache[index][0].to(self.device, non_blocking=True)
            self._cache[index][1] = self._cache[index][1].to(self.device, non_blocking=True)
        return self._cache[index]
    
    # TODO: correct these for filtering indices
    # def _get_up_frequencies(self) -> List[float]:
    #     not_binary_indices = return_not_binary_indices(self.ds)
    #     binary_indices = list(range(self.n_features))[len(not_binary_indices):]
    #     up_frequencies = np.array([np.array(np.count_nonzero(
    #         self.ds[:, i])/len(self.ds)) for i in binary_indices])
    #     return up_frequencies
    
    # def _get_binary_indices(self) -> List[int]:
    #     not_binary_indices = return_not_binary_indices(self.ds)
    #     binary_indices = list(range(self.n_features))[len(not_binary_indices):]
    #     return binary_indices
    
    # def _get_not_binary_indices(self) -> List[int]:
    #     not_binary_indices = return_not_binary_indices(self.ds)
    #     return not_binary_indices


    def __len__(self) -> int:
        return len(self.indices)
