import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
                           bias=False
                           )

        stacked_v_r_dim = 2

        self.w_from_i2q = nn.Parameter(
            torch.zeros(q_ch, 1, 3, 3), requires_grad=True)

        self.w_from_v2q = nn.Parameter(
            torch.zeros(q_ch, 1, 3, 3), requires_grad=True)

        self.q = nn.Conv2d(in_channels=stacked_v_r_dim,
                           out_channels=q_ch,
                           kernel_size=3,
                           padding=3//2,
                           bias=False)

        self.q.weight.data = torch.cat([self.w_from_i2q, self.w_from_v2q], 1)

        self.fc = nn.Linear(in_features=q_ch,
                            out_features=n_act,
                            bias=False)
    def forward(self, x, k):
        s1, s2, obs = x

        r_img = self.h(obs)

        r = self.r(r_img)
        q =  F.conv2d(r,
                self.w_from_i2q,
                stride=1,
                padding=1)

        v, _ = torch.max(q, 1)

        v = v.unsqueeze(1)

        for _ in range(k):
            rv = torch.cat([r,v], 1)
            q = self.q(rv)

            v, _ = torch.max(q, 1)
            v = v.unsqueeze(1)

        rv = torch.cat([r, v], 1)
        q = self.q(rv)

        slice_s1 = s1.long().expand(obs.shape[-1], 1, q.shape[1], q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze()

        slice_s2 = s2.long().expand(1, q.shape[1], q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_att = q_out.gather(2, slice_s2).squeeze()

        logits = self.fc(q_att)


        return logits

