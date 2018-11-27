import torch

import torch.nn as nn

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

        self.q = nn.Conv2d(in_channels=stacked_v_r_dim,
                           out_channels=q_ch,
                           kernel_size=3,
                           padding=3//2,
                           bias=False)

        self.fc = nn.Linear(in_features=q_ch,
                            out_features=n_act,
                            bias=False)

    def forward(self, x, k):
        s1, s2, obs = x

        r_img = self.h(obs)
        r = self.r(r_img)

        v = torch.zeros(r.size()).cuda()

        for _ in range(k):
            rv = torch.cat([r, v], 1)
            q = self.q(rv)
            v, _ = torch.max(q, 1)
            v = v.view((v.shape[0], 1, v.shape[1], v.shape[2]))

        rv = torch.cat([r, v], 1)
        q = self.q(rv)

        slice_s1 = s1.long().expand(obs.shape[-1], 1, q.shape[1], q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)

        slice_s2 = s2.long().expand(1, q.shape[1], q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q = q_out.gather(2, slice_s2).squeeze(2)


        logits = self.fc(q)


        return logits

