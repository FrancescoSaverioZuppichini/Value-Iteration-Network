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
import pandas as pd

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WORLD_8X8 = './data/gridworld_8x8.npz', (8, 8)
WORLD_16X16 = './data/gridworld_16x16.npz', (16, 16)
WORLD_28X28 = './data/gridworld_28x28.npz', (28, 28)

k_zoo = {
    WORLD_8X8: 10,
    WORLD_16X16: 20,
    WORLD_28X28: 30
}

TRAIN = True
EPOCHES = 30

world = WORLD_8X8

world_name, _ = path.splitext(path.basename(world[0]))
save_path = 'model-{}.pt'.format(world_name)

train_ds = GridWorldDataset(*world, train=True)
test_ds = GridWorldDataset(*world, train=False)

train_dl = DataLoader(dataset=train_ds,
                      batch_size=128,
                      num_workers=14,
                      pin_memory=True,
                      shuffle=True)

test_dl = DataLoader(dataset=test_ds,
                     batch_size=128,
                     num_workers=14,
                     pin_memory=True,
                     shuffle=False)

print(device)
print('Train size={}, Test size={}'.format(len(train_dl), len(test_dl)))

vin = VIN(in_ch=2, n_act=8).to(device)

optimizer = optim.RMSprop(vin.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

print(vin)

def run(dl, epoches, k, train=True):
    losses = []
    accuracies = []
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

        tot_loss = (tot_loss / n_batch).item()
        tot_acc = (tot_acc / n_batch).item()
        print('{} loss={:.4f} acc={:.4f}'.format(epoch,
                                                 tot_loss,
                                                 tot_acc))
        losses.append(tot_loss)
        accuracies.append(tot_acc)

    return losses, accuracies


k = k_zoo[world]

if TRAIN:
    train_loss, train_acc = run(train_dl, EPOCHES, k=k)
    torch.save(vin, save_path)

    vin = torch.load(save_path)

    test_loss, test_acc = run(test_dl, 1, train=False, k=k)

    df = pd.DataFrame(data={ 'train_loss': train_loss,
                             'train_acc': train_acc})

    df.to_csv('./{}/train.csv'.format(world_name))

    df = pd.DataFrame(data={'test_loss': test_loss,
                            'test_acc': test_acc})
    df.to_csv('./{}/test.csv'.format(world_name))


vin = torch.load(save_path)

labels, s1, s2, obs = get_random_data(test_ds, device, idx=1)

_, v, r_img = vin((s1, s2, obs), 10)

os.makedirs('./' + world_name, exist_ok=True)

for i, img in enumerate(vin.values):
    fig = plt.figure()
    sns.heatmap(img)
    fig.savefig('./{}/{}.png'.format(world_name, i))
    plt.close(fig)

fig = make_images(obs, r_img, v)
fig.savefig('./{}/figures.png'.format(world_name))
