import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from scipy.io import loadmat


class GridWorldDataset(Dataset):
    def __init__(self, path, img_size):
        self.img_size = img_size
        self.labels, self.s1, self.s2, self.images, = self.get_data(path)

    def get_data(self, path):

        with np.load(path) as f:
            data = f.items()[0][1][0]

        labels, s1, s2, images = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        images = images.reshape((-1, 1, *self.img_size))

        labels, s1, s2, images = torch.from_numpy(labels), \
                                 torch.from_numpy(s1).int(), \
                                 torch.from_numpy(s2).int(), \
                                 torch.from_numpy(images),
        return labels, s1, s2, images

    def __getitem__(self, idx):
        return self.labels[idx], self.s1[idx], self.s2[idx], self.images[idx]

    def __len__(self):
        return self.images.shape[0]

ds = GridWorldDataset('./data/gridworld_8x8.npz', (8,8))

print(len(ds))
for i in range(len(ds)):
    label, s1, s2, image = ds[i]

    print(image)

    # plt.imshow(np.transpose(image.numpy(),[2, 1, 0]).squeeze())
    # plt.show()