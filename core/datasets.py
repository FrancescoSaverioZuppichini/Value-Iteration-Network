import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from scipy.io import loadmat


class GridWorldDataset(Dataset):
    def __init__(self, path, img_size):
        self.img_size = img_size
        self.labels, self.s1, self.s2, self.obs, = self.get_data(path)

    def get_data(self, path):

        with np.load(path) as f:
            data = list(f.items())[0][1][0]

        labels, s1, s2, obs = data[:, 0], data[:, 1], data[:, 2], data[:, 3:]
        # obs has two channels. One with 1 if obstacles and 0 otherwise, one with 1 if goal 0 otherwise
        obs = obs.reshape((-1, self.img_size[0], self.img_size[1], 2)).transpose((0, 3, 1, 2))

        labels, s1, s2, images = torch.from_numpy(labels), \
                                 torch.from_numpy(s1).int(), \
                                 torch.from_numpy(s2).int(), \
                                 torch.from_numpy(obs),
        return labels, s1, s2, images

    def __getitem__(self, idx):
        return self.labels[idx], self.s1[idx], self.s2[idx], self.obs[idx]

    def __len__(self):
        return self.obs.shape[0]