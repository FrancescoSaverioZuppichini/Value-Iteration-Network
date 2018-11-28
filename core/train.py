import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from os import path

from core.model import VIN
from core.datasets import GridWorldDataset

from core.utils import *

torch.manual_seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WORLD_8X8 = './data/gridworld_8x8.npz', (8,8)
WORLD_16X16 = './data/gridworld_16x16.npz', (16,16)

world = WORLD_16X16

world_name, _ = path.splitext(path.basename(world[0]))
save_path = 'model-{}.pt'.format(world_name)

TRAIN = True

train_ds = GridWorldDataset(*world, train=True)
test_ds = GridWorldDataset(*world, train=False)

train_dl = DataLoader(dataset=train_ds,
                batch_size=256,
                num_workers=14,
                pin_memory=True,
                shuffle=True)

test_dl = DataLoader(dataset=test_ds,
                batch_size=256,
                num_workers=14,
                pin_memory=True,
                shuffle=False)

print(device)
print('Train size={}, Test size={}'.format(len(train_dl), len(test_dl)))
vin = VIN(in_ch=2, n_act=8).to(device)

optimizer = optim.RMSprop(vin.parameters(), lr=0.002, eps=1e-6)

criterion = nn.CrossEntropyLoss()

print(vin)

def run(dl, epoches, train=True):

    for epoch in range(epoches):
        tot_loss = torch.zeros(1).to(device)
        tot_acc = torch.zeros(1).to(device)

        for n_batch, (labels, s1, s2, obs) in enumerate(dl):
            labels, s1, s2, obs = labels.to(device).long(), \
                                       s1.to(device), \
                                       s2.to(device), \
                                  obs.to(device)
            if train: optimizer.zero_grad()

            outputs, _ = vin((s1, s2, obs), k=10)

            loss = criterion(outputs.to(device), labels)

            if train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, 1)

                acc = (labels == preds).sum().float() / labels.shape[0]

                tot_loss += loss
                tot_acc += acc

        print('{} loss={:.4f} acc={:.2f}'.format(epoch,
                                                 (tot_loss/ n_batch).item(),
                                                 (tot_acc / n_batch).item()))

if TRAIN:
    print('Train')
    run(train_dl, 30)
    torch.save(vin, save_path)

    run(test_dl, 1, train=False)

vin = torch.load(save_path)

print('Test')
run(test_dl, 1, train=False)

test = test_ds[0]
(labels, s1, s2, obs) = test

labels, s1, s2, obs = labels.unsqueeze(0).to(device).long(), \
                      s1.unsqueeze(0).to(device), \
                      s2.unsqueeze(0).to(device), \
                      obs.unsqueeze(0).to(device)



_, v = vin((s1, s2, obs), 10)

torch_imshow(obs[0][0].squeeze(), 'world')
torch_imshow(obs[0][1].squeeze(), 'r')

torch_imshow(v[0], 'v')

# fig = plt.figure()
# plt.title('r_img')
# img = r[0].detach().squeeze().cpu().numpy()
# plt.imshow(img)
# fig.show()

# v, _ = torch.max(q, 1)
#
# fig = plt.figure()
# plt.title('r')
# img = obs[0][0].detach().squeeze().cpu().numpy()
# plt.imshow(img)
# fig.show()
#
# fig = plt.figure()
# plt.title('grid')
# plt.imshow(obs[0][1].detach().squeeze().cpu().numpy())
# fig.show()
#
# fig = plt.figure()
# v = v.detach().squeeze().cpu().numpy()
# plt.title('v')
# plt.imshow(v)
# fig.show()
#     print(label, s1, s2, image)
# print(len(ds))