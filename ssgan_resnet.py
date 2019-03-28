# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np


channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = []
        if in_channels != out_channels:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, 1.)
            self.bypass.append(SpectralNorm(self.bypass_conv))
        if stride != 1:
            self.bypass.append(nn.Upsample(scale_factor=2))
        self.bypass = nn.Sequential(*self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = []
        if in_channels != out_channels:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, 1.)
            self.bypass.append(SpectralNorm(self.bypass_conv))
        if stride != 1:
            self.bypass.append(nn.AvgPool2d(2, stride=stride, padding=0))
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )
        self.bypass = nn.Sequential(*self.bypass)


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

#GEN_SIZE=128
#DISC_SIZE=128
CH = 64 

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4*4*8*CH)
        self.final = nn.Conv2d(CH, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(8*CH, 8*CH, stride=2),
            ResBlockGenerator(8*CH, 4*CH, stride=2),
            ResBlockGenerator(4*CH, 2*CH, stride=2),
            ResBlockGenerator(2*CH, CH, stride=2),
            nn.BatchNorm2d(CH),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, 8*CH, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, 1*CH, stride=2),
                ResBlockDiscriminator(1*CH, 2*CH, stride=2),
                ResBlockDiscriminator(2*CH, 4*CH, stride=2),
                ResBlockDiscriminator(4*CH, 8*CH, stride=2),
                ResBlockDiscriminator(8*CH, 8*CH, stride=2),
                nn.ReLU(),
                nn.AvgPool2d(4),
            )
        self.fc = nn.Linear(8*CH, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

        self.rotate_fc = nn.Linear(8*CH, 4)
        nn.init.xavier_uniform(self.rotate_fc.weight.data, 1.)
        self.rotate_fc = SpectralNorm(self.rotate_fc)
        

    def forward(self, x):
        feat = self.model(x).view(-1, 8*CH)
        disc_score = self.fc(feat)
        rotate_score = self.rotate_fc(feat)
        return disc_score, rotate_score
        #return self.fc(self.model(x).view(-1,DISC_SIZE))
