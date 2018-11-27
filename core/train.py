import torch

import torch.nn as nn
import torch.optim as optim

from core.model import VIN
from core.datasets import GridWorldDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ds = GridWorldDataset('./data/gridworld_8x8.npz', (8,8))
dl = DataLoader(dataset=ds,
                batch_size=128,
                num_workers=14,
                pin_memory=True,
                drop_last=True)


print(device)

vin = VIN(in_ch=2, n_act=8).to(device)

optimizer = optim.RMSprop(vin.parameters(), lr=0.02)

criterion = nn.CrossEntropyLoss()
print(vin)
#
for i, (labels, s1, s2, images) in enumerate(dl):
    labels, s1, s2, images = labels.to(device), \
                               s1.to(device), \
                               s2.to(device), \
                               images.to(device)
    optimizer.zero_grad()

    outputs = vin((s1, s2, images), k=10)

    loss = criterion(outputs.to(device), labels.long())

    print(loss)
    optimizer.step()
#
#     print(label, s1, s2, image)
# print(len(ds))