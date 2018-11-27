import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from scipy.io import loadmat


class GridWorldDataset(Dataset):
    def __init__(self, path, img_size):
        self.img_size = img_size
        self.labels, self.s1, self.s2, self.images, = self.get_data(path)

    def get_data(self, path):

        with np.load(path) as f:
            data = list(f.items())[0][1][0]

        labels, s1, s2, images = data[:, 0], data[:, 1], data[:, 2], data[:, 3:]

        images = images.reshape((-1, self.img_size[0], self.img_size[1], 2)).transpose((0, 3, 1, 2))

        labels, s1, s2, images = torch.from_numpy(labels), \
                                 torch.from_numpy(s1).int(), \
                                 torch.from_numpy(s2).int(), \
                                 torch.from_numpy(images),
        return labels, s1, s2, images

    def __getitem__(self, idx):
        return self.labels[idx], self.s1[idx], self.s2[idx], self.images[idx]

    def __len__(self):
        return self.images.shape[0]
