from typing import Union
from pathlib import Path
import numpy as np
import json
import torch
from torch.utils.data import Dataset

class CachedFEMNIST(Dataset):
    def __init__(
        self,
        client_id: int,
        train: bool = True,
        device: str = 'cpu',
        seed: int = 51550,
        path_to_data: Path = None,
        ):
        clients_list = path_to_data.glob('{}/*.json'.format(
            'train' if train else 'test'
        ))
        n_clients = len(clients_list)
        if client_id < 0 or client_id > n_clients-1:
            raise ValueError(
                'client_id must be in the range [0, (n_clients-1)], was given {} with n_clients {}'.
                format(client_id, n_clients))
        np.random.default_rng(seed)
        np.random.shuffle(clients_list)
        with open(clients_list[client_id], 'r') as file:
            json_file = json.load(file)
            self.x = np.array(json_file['x']).squeeze()
            self.y = np.array(json_file['y']).squeeze()
        self.device = device
        self._cache = dict()
        

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = [list(self.x[index]), self.y]
            self._cache[index][0] = self._cache[index][0].to(self.device, non_blocking=True)
            self._cache[index][1] = torch.tensor(self._cache[index][1]).to(
                self.device,
                non_blocking=True
            )
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.y)
