import torch
from torch import nn, optim
import argparse
import numpy as np
from PIL import Image as im

from dataset import NPYDataset
from model import Network

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)


model = Network(args)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss(reduction='sum')

train_dataset = NPYDataset("data/full-numpy_bitmap-apple.npy", train=True)
test_dataset = NPYDataset("data/full-numpy_bitmap-apple.npy", train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    sampler=None,
    drop_last=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    sampler=None,
    drop_last=True)

train_loss = 0

def train(model, data_loader, loss_fn, opt):
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device).float()
            opt.zero_grad()
            output = model(data) * 255
            loss = loss_fn(output, data)
            print(loss.item())
            loss.backward()
            opt.step()

def test(model, data_loader, loss_fn):
    model.eval()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device).float()
        output = model(data) * 255
        loss = loss_fn(output, data)
        print(loss.item())

train(model, train_loader, loss_fn, opt)
test(model, test_loader, loss_fn)
