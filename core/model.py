import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.ion()

def attention(x):

    s1, s2, obs, q = x
    slice_s1 = s1.long().expand(obs.shape[-1], 1, q.shape[1], q.shape[0])
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = q.gather(2, slice_s1).squeeze()
    #  problem with squeeze when batch size = 1
    if len(q_out.shape) == 2: q_out = q_out.unsqueeze(0)

    slice_s2 = s2.long().expand(1, q.shape[1], q.shape[0])
    slice_s2 = slice_s2.permute(2, 1, 0)

    x = q_out.gather(2, slice_s2).squeeze()

    return x

class q(nn.Module):
    def __init__(self, q_ch):
        super().__init__()
        self.w_from_i2q = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(q_ch, 1, 3, 3)))

        self.w_from_v2q = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(q_ch, 1, 3, 3)))

    def forward(self, x):
        if x.shape[1] == 1:
            x = F.conv2d(x,
                     self.w_from_i2q,
                     stride=1,
                     padding=1)
        else:
            x = F.conv2d(
                x,
                torch.cat([self.w_from_i2q, self.w_from_v2q], 1),
                stride=1,
                padding=1)

        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class VIN(nn.Module):
    def __init__(self, in_ch, n_act, h_ch=150, r_ch=1, q_ch=10):
        super().__init__()

        self.h = nn.Conv2d(in_channels=in_ch,
                           out_channels=h_ch,
                           kernel_size=3,
                           padding=3//2,
                           bias=True)

        self.r = nn.Conv2d(in_channels=h_ch,
                           out_channels=r_ch,
                           kernel_size=3,
                           padding=3//2,
                           bias=False,
                           )

        self.q = q(q_ch)

        self.fc = nn.Linear(in_features=q_ch,
                            out_features=n_act,
                            bias=False)

        self.apply(weights_init)


    def forward(self, x, k, store=False):
        s1, s2, obs = x
        self.values = []
        r_img = self.h(obs)

        r = self.r(r_img)
        q = self.q(r)

        for _ in range(k + 1): # include last iteration
            v, _ = torch.max(q, 1)
            self.values.append(v[0].detach().squeeze().cpu().numpy())
            v = v.unsqueeze(1)
            q = self.q(torch.cat([r,v], 1))


        q_att = attention((s1, s2, obs, q))

        logits = self.fc(q_att)


        return logits, v, r

