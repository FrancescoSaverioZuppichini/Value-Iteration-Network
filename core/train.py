import torch

import torch.nn as nn
import torch.optim as optim

import os
import cv2
import seaborn as sns
import numpy as np

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from os import path

from core.model import VIN
from core.datasets import GridWorldDataset

from core.utils import *

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WORLD_8X8 = './data/gridworld_8x8.npz', (8, 8)
WORLD_16X16 = './data/gridworld_16x16.npz', (16, 16)
WORLD_28X28 = './data/gridworld_28x28.npz', (28, 28)

world = WORLD_28X28

world_name, _ = path.splitext(path.basename(world[0]))
save_path = 'model-{}.pt'.format(world_name)

train_ds = GridWorldDataset(*world, train=True)
test_ds = GridWorldDataset(*world, train=False)

train_dl = DataLoader(dataset=train_ds,
                      batch_size=512,
                      num_workers=14,
                      pin_memory=True,
                      shuffle=True)

test_dl = DataLoader(dataset=test_ds,
                     batch_size=512,
                     num_workers=14,
                     pin_memory=True,
                     shuffle=False)

print(device)
print('Train size={}, Test size={}'.format(len(train_dl), len(test_dl)))

vin = VIN(in_ch=2, n_act=8).to(device)

optimizer = optim.RMSprop(vin.parameters(), lr=0.005)

criterion = nn.CrossEntropyLoss()

print(vin)

def run(dl, epoches, k, train=True):
    for epoch in range(epoches):
        tot_loss = torch.zeros(1).to(device)
        tot_acc = torch.zeros(1).to(device)

        for n_batch, (labels, s1, s2, obs) in enumerate(dl):
            labels, s1, s2, obs = labels.to(device).long(), \
                                  s1.to(device), \
                                  s2.to(device), \
                                  obs.to(device)

            if train: optimizer.zero_grad()

            outputs, v, _ = vin((s1, s2, obs), k=k)

            loss = criterion(outputs.to(device), labels)

            if train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, 1)

                acc = (labels == preds).sum().float() / labels.shape[0]

                tot_loss += loss
                tot_acc += acc


        print('{} loss={:.4f} acc={:.4f}'.format(epoch,
                                                 (tot_loss / n_batch).item(),
                                                 (tot_acc / n_batch).item()))


TRAIN = True
k = 30
if TRAIN:
    run(train_dl, 30, k=k)
    torch.save(vin, save_path)

vin = torch.load(save_path)

run(test_dl, 1, train=False, k=k)

labels, s1, s2, obs = get_random_data(test_ds, device, idx=0)

_, v, r_img = vin((s1, s2, obs), k)

os.makedirs('./' + world_name, exist_ok=True)

for i, img in enumerate(vin.values):
    fig = plt.figure()
    sns.heatmap(img)
    fig.savefig('./{}/{}.png'.format(world_name, i))
    plt.close(fig)

fig = make_images(obs, r_img, v)
fig.savefig('./{}/figures.png'.format(world_name))
