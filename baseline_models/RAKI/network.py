import torch
import torch.nn as nn
import torch.nn.init as init

class RAKI(nn.Module):
    def __init__(self, R, nc=32):
        super(RAKI, self).__init__()

        # network parameters
        self.R = R
        self.nc = nc
        self.b1 = (5, 2)
        self.n1 = 32
        self.b2 = (1, 1)
        self.n2 = 8
        self.b3 = (3, 2)
        self.no = (self.R - 1)
        self.dilation = (1, self.R)

        torch.manual_seed(18)
        

        self.CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.nc,
                out_channels=self.n1,
                kernel_size=self.b1,
                bias=False,
                padding="valid",
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.n1,
                out_channels=self.n2,
                kernel_size=self.b2,
                bias=False,
                padding="valid",
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.n2,
                out_channels=self.no,
                kernel_size=self.b3,
                bias=False,
                padding="valid",
                dilation=self.dilation,
            ),
        )
        self.initialize_weights_bias()

    def forward(self, x):
        x = self.CNN(x)
        return x

    def initialize_weights_bias(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                init.normal_(
                    m.weight, mean=0, std=0.1
                )

