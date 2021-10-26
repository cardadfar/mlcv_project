import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image as im

from dataset import NPYDataset
from model import Network

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

print(args)

model = Network(args)
model.to(device) 

opt = optim.Adam(model.parameters(), lr=1e-4)
#loss_fn = nn.MSELoss(reduction='sum')
loss_fn = nn.BCELoss()

classList = ['apple']
#classList = None
train_dataset = NPYDataset("data/", train=True, classList=classList)
test_dataset = NPYDataset("data/", train=False, classList=classList)

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

def save_model(model, loss_fn, opt, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss_fn
        }, "checkpoints/model_epoch_" + str(epoch))
    
    print("Model saved at epoch {0}.".format(epoch))

def load_model(model, loss_fn, opt, epoch):
    checkpoint = torch.load("checkpoints/model_epoch_" + str(epoch), map_location=device)
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
            output = model(data) #* 255
            

            data = data / 255.0
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
        true.save('plots/train/true.png')
        pred_data = (output[0] * 255).detach().cpu().numpy()
        pred = im.fromarray(np.uint8(pred_data))
        pred.save('plots/train/pred_epoch_' + str(epoch) + '.png')


def sigmoid(x, scale=15.0):
    return 1.0 / (1.0 + np.exp(scale * (-x + 0.5)))


def iden(x):
    return x

def save_img(data, name, test=True):
    true_data = data.detach().cpu().numpy()
    true = im.fromarray(np.uint8(true_data))
    if test:
        true.save('plots/test/' + name)
    else:
        true.save('plots/train/' + name)

def B(x, k, i, t):
    if k == 0:
       return 1.0 if torch.all((t[i] <= x) == True) and torch.all((x < t[i+1]) == True) else 0.0
    c1 = (x - t[i])/(t[i+k] - t[i] + EPS_F) * B(x, k-1, i, t)
    c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1] + EPS_F) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline_interp(x, t, c, k):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))

def bspline(x, ibf=1, ease=iden):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = ease(f / (ibf + 2.0))
            xt = linear_interp(x[i], x[i+1], t)
            k = 2
            c = np.zeros(n - k - 1)
            c[0] = -1
            c[-1] = 1
            ibtwn = bspline_interp(xt, x, c, k)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    return y

def catmullRom(x, ibf=1, ease=iden):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = ease(f / (ibf + 2.0))
            if i == 0:
                ibtwn = catmullRom_interp(x[i], x[i], x[i+1], x[i+2], t)
            elif i == n - 2:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+1], t)
            else:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+2], t)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    return y

def linear_interp(x0, x1, t):
    return (1.0 - t) * x0 + t * x1

def catmullRom_interp(x0, x1, x2, x3, t):
    # https://www.mvps.org/directx/articles/catmull/
    return  0.5 * ((2*x1) +\
                    (-x0 + x2) * t +\
                    (2*x0 - 5*x1 + 4*x2 - x3) * t**2 +\
                    (-x0 + 3*x1- 3*x2 + x3) * t**3)

def linear(x, ibf=1, ease=iden):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = ease(f / (ibf + 2.0))
            ibtwn = linear_interp(x[i], x[i+1], t)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    return y


def test(model, data_loader, loss_fn):
    model.eval()
    avg_loss = 0 
    avg_loss_cnt = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device).float()
        output = model(data)

        loss = loss_fn(output, data / 255.0)
        avg_loss += loss.item()
        avg_loss_cnt += 1

        if batch_idx == 0:
            encoded = model.encode(data[0:10].view(-1, 784))

            #y = linear(encoded, ibf=5, ease=sigmoid)
            y = catmullRom(encoded, ibf=5, ease=iden)
            #y = bspline(encoded, ibf=5, ease=iden)
            y = y.to(device)
            output = model.decode(y)
            output = torch.sigmoid(8*(output-0.5)) * 255

            for i in range(len(output)):
                save_img(output[i], str(i) + '.png')
            
            return


            
    print('Epoch (Test): [{0}]\t'
        'Loss {1:.2f}\t'.format(
                        epoch,
                        avg_loss / avg_loss_cnt))

epoch = 0
model, loss_fn, opt, epoch = load_model(model, loss_fn, opt, 5)
test(model, test_loader, loss_fn)
#train(model, train_loader, loss_fn, opt, epoch)

