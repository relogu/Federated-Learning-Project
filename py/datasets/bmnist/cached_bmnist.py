import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

class CachedBMNIST(Dataset):
    def __init__(self, train, device, path=None, download=True, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST("./data" if path is None else path,
                        download=download,
                        train=train,
                        transform=img_transform)
        self.device = device
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (
            torch.round(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float().div(255))
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            self._cache[index][0] = self._cache[index][0].to(self.device, non_blocking=True)
            self._cache[index][1] = torch.tensor(self._cache[index][1]).to(
                self.device,
                non_blocking=True
            )
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)
