import torch
from torch.utils.data import Dataset

class ReconstructedBMNIST(Dataset):
    def __init__(self, reconstructed_imgs, actual, cuda):
        self.r_imgs = torch.round(reconstructed_imgs)
        self.actual = actual
        self.cuda = cuda

    def __getitem__(self, index: int) -> torch.Tensor:
        r_img = self.r_imgs[index]
        label = self.actual[index]
        if self.cuda:
            r_img.cuda(non_blocking=True)
            label.cuda(non_blocking=True)
        return [r_img, label]

    def __len__(self) -> int:
        return len(self.ds)
