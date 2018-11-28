import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from core.model import VIN
from core.datasets import GridWorldDataset

torch.manual_seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

WORLD_8X8 = './data/gridworld_8x8.npz', (8,8)
WORLD_16X16 = './data/gridworld_16x16.npz', (16,16)

train_ds = GridWorldDataset(*WORLD_16X16, train=True)
test_ds = GridWorldDataset(*WORLD_16X16, train=False)

train_dl = DataLoader(dataset=train_ds,
                batch_size=256,
                num_workers=14,
                pin_memory=True,
                drop_last=True,
                shuffle=True)

test_dl = DataLoader(dataset=test_ds,
                batch_size=256,
                num_workers=14,
                pin_memory=True,
                drop_last=True,
                shuffle=False)

print(device)
print('Train size={}, Test size={}'.format(len(train_dl), len(test_dl)))
vin = VIN(in_ch=2, n_act=8).to(device)

optimizer = optim.Adam(vin.parameters(), lr=0.005)

criterion = nn.CrossEntropyLoss()

print(vin)

def run(dl, epoches):

    for epoch in range(epoches):
        tot_loss = torch.zeros(1).to(device)
        tot_acc = torch.zeros(1).to(device)

        for n_batch, (labels, s1, s2, obs) in enumerate(dl):
            labels, s1, s2, obs = labels.to(device).long(), \
                                       s1.to(device), \
                                       s2.to(device), \
                                  obs.to(device)
            optimizer.zero_grad()

            outputs, _ = vin((s1, s2, obs), k=20)

            loss = criterion(outputs.to(device), labels)

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

run(train_dl, 30)
run(test_dl, 1)
#     print(label, s1, s2, image)
# print(len(ds))