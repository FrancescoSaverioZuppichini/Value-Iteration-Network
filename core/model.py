import torch

import torch.nn as nn

class VIN(nn.Module):
    def __init__(self, in_ch, n_act, h_ch=150, r_ch=1, q_ch=1):
        super().__init__()

        self.h = nn.Conv2d(in_channels=in_ch,
                           out_channels=h_ch,
                           kernel_size=3,
                           padding=3//2)

        self.r = nn.Conv2d(in_channels=h_ch,
                           out_channels=r_ch,
                           kernel_size=3,
                           padding=3//2
                           )

        stacked_v_r_dim = 2

        self.q = nn.Conv2d(in_channels=stacked_v_r_dim,
                           out_channels=q_ch,
                           kernel_size=3,
                           padding=3//2)

        self.fc = nn.Linear(in_features=q_ch,
                            out_features=n_act,
                            bias=False)

        self.v = nn.Parameter(torch.zeros(r_ch))

    def forward(self, x, k):
        s1, s2, image = x

        r_img = self.h(image)


        return torch.zeros((image.shape[0], 8))
