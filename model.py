import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.channel_mult*1, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*10, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*10, self.channel_mult*12, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*12, self.channel_mult*16, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))


    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.input_size = input_size
        self.channel_mult = 16
        self.output_channels = 1
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*12,
                                4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*12),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*12, self.channel_mult*8,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*6,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*6),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*6, self.channel_mult*4,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, *self.input_size)


class Network(nn.Module):
    def __init__(self, args, input_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.encoder = CNN_Encoder(args.embedding_size, input_size)
        self.decoder = CNN_Decoder(args.embedding_size, input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self, args, input_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.conv = CNN_Encoder(64, input_size)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.conv(x.view(-1, *self.input_size))
        z = self.fc(z)
        return torch.sigmoid(z)
