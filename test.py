import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image as im
import os

from torch.nn.modules.loss import BCEWithLogitsLoss

from dataset import TestDataset
from model import Network, Discriminator
from interp import *
from util import img2vid, save2json


if __name__ == '__main__':

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
    parser.add_argument('--img-class', type=str, default='apple', metavar='N',
                        help='class to use')
    parser.add_argument('--print-freq', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-freq', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
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
    # torch.manual_seed(args.seed)

    print(args)

    MODEL_TYPE = args.img_class

    def load_model(model, loss_fn, opt, epoch):
        checkpoint = torch.load("checkpoints/" + MODEL_TYPE + "/model_epoch_5", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_fn = checkpoint['loss']

        print("Model loaded at epoch {0}.".format(epoch))

        return model, loss_fn, opt, epoch

    def save_img(data, name, test=True):
        # data = torch.sigmoid(8 * (data - 0.5))
        true_data = data[0].detach().cpu().numpy() * 255
        true = im.fromarray(np.uint8(true_data))
        if test:
            true.save(args.results_path + 'test/' + name)
        else:
            true.save(args.results_path + 'train/' + name)


    def test(model, data_loader, loss_fn, epoch=0):
        model.eval()
        for batch_idx, (imgs, _, _) in enumerate(data_loader):
            imgs = imgs.to(device).float()

            if batch_idx == 0:
                print(imgs.shape)
                encoded = model.encode(imgs)
                print(encoded.shape)

                y = linear(encoded, ibf=20, ease=sigmoid)
                #y = catmullRom(encoded, ibf=20, ease=sigmoid)
                y = y.to(device)

                output = model.decode(y)
                output = torch.sigmoid(8 * (output - 0.5))
                output[output > 0.6] = 1.0
                output[output < 0.3] = 0.0
                # output = torch.where(output > 0.5, 1.0, torch.where(output > 0.3, 0.5, 0.0).double())
                
                for i in range(len(output)):
                    save_img(output[i], str(i) + '.png')
                

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.results_path + 'test/', exist_ok=True)

    test_dataset = TestDataset('data/png/' + MODEL_TYPE)

    model = Network(args, input_size=(1, 256, 256))
    model.to(device) 
    
    discriminator = Discriminator(args, input_size=(1, 256, 256))
    discriminator.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True)
        
    model, loss_fn, opt, epoch = load_model(model, loss_fn, opt, 20)
    test(model, test_loader, loss_fn, 0)
