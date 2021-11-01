import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image as im
import os

from dataset import SketchDataset
from model import Network
from interp import *

EPS_F = 1e-7

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
parser.add_argument('--print-freq', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data-path', type=str, default='data/png/', metavar='N',
                    help='Where to load images')
parser.add_argument('--results-path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--checkpoint-path', type=str, default='checkpoints/', metavar='N',
                    help='Where to store models')
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

print(args)

os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.results_path + 'train/', exist_ok=True)
os.makedirs(args.results_path + 'test/', exist_ok=True)

model = Network(args, input_size=(1, 256, 256))
model.to(device) 

opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-1)
#loss_fn = nn.MSELoss(reduction='sum')
loss_fn = nn.BCELoss()

class_list = ['angel']
# classList = None
train_dataset = SketchDataset(args.data_path, train=True, class_list=class_list)
test_dataset = SketchDataset(args.data_path, train=False, class_list=class_list)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
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

def save_model(model, loss_fn, opt, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss_fn
        }, args.checkpoint_path + "model_epoch_" + str(epoch))
    
    print("Model saved at epoch {0}.".format(epoch))

def load_model(model, loss_fn, opt, epoch):
    checkpoint = torch.load(args.checkpoint_path + "model_epoch_" + str(epoch), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_fn = checkpoint['loss']

    print("Model loaded at epoch {0}.".format(epoch))

    return model, loss_fn, opt, epoch

def train(model, data_loader, loss_fn, opt, epoch):
    model.train()
    while epoch < args.epochs:
        avg_loss = 0
        avg_loss_cnt = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device).float()
            opt.zero_grad()
            output = model(data)
            
            data = data
            loss = loss_fn(output, data)
            avg_loss += loss.item()
            avg_loss_cnt += 1

            if batch_idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {3:.2f} ({4:.2f})\t'.format(
                      epoch,
                      batch_idx,
                      len(train_loader),
                      loss.item(),
                      avg_loss / avg_loss_cnt))

            loss.backward()
            opt.step()

        epoch += 1

        if epoch % args.save_freq == 0:
            save_model(model, loss_fn, opt, epoch)

        true_data = (data[0] * 255).detach().cpu().numpy()
        true = im.fromarray(np.uint8(true_data))
        true.save(args.results_path + 'train/true_epoch_' + str(epoch) + '.png')
        pred_data = (output[0] * 255).detach().cpu().numpy()
        pred = im.fromarray(np.uint8(pred_data))
        pred.save(args.results_path + 'train/pred_epoch_' + str(epoch) + '.png')

def save_img(data, name, test=True):
    true_data = data.detach().cpu().numpy()
    true = im.fromarray(np.uint8(true_data))
    if test:
        true.save(args.results_path + 'test/' + name)
    else:
        true.save(args.results_path + 'train/' + name)

def test(model, data_loader, loss_fn, epoch=0):
    model.eval()
    avg_loss = 0 
    avg_loss_cnt = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device).float()
        output = model(data)

        loss = loss_fn(output, data)
        avg_loss += loss.item()
        avg_loss_cnt += 1

        if batch_idx == 0:
            idx = np.random.choice(len(data), 20, replace=False)
            encoded = model.encode(data[idx].view(-1, 256 * 256))

            #y = linear(encoded, ibf=5, ease=sigmoid)
            y = catmullRom(encoded, ibf=20, ease=iden)
            #y = bspline(encoded, ibf=5, ease=iden)
            y = y.to(device)
            output = model.decode(y) * 255

            for i in range(len(output)):
                save_img(output[i], str(i) + '.png')
            
            return
            
    print('Epoch (Test): [{0}]\t'
        'Loss {1:.2f}\t'.format(
                        epoch,
                        avg_loss / avg_loss_cnt))

# model, loss_fn, opt, epoch = load_model(model, loss_fn, opt, 5)
# test(model, test_loader, loss_fn)
train(model, train_loader, loss_fn, opt, 0)
test(model, test_loader, loss_fn, 0)
