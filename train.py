import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image as im
import os

from torch.nn.modules.loss import BCEWithLogitsLoss

from dataset import SketchDataset
from model import Network, Discriminator
from interp import *
from util import save2json


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


    def train(model, discriminator, train_loader, test_loader, loss_fn, opt, opt_d, epoch):
        model.train()
        while epoch < args.epochs:
            avg_loss = 0
            avg_loss_cnt = 0
            for batch_idx, (imgs, _, _) in enumerate(train_loader):
                imgs = imgs.to(device).float()
                
                emb = model.encode(imgs.view(-1, *model.input_size))
                output = model.decode(emb)
                
                loss = loss_fn(output, imgs)
                avg_loss += loss.item()
                avg_loss_cnt += 1

                idxs = np.random.randint(len(imgs), size=[len(imgs), 2])
                zs = list()
                for (i1, i2) in idxs:
                    alpha = np.random.uniform(0.2, 0.8)
                    z = alpha * emb[i1] + (1 - alpha) * emb[i2]
                    zs.append(z)

                zs = torch.stack(zs, 0)
                rec = model.decode(zs)

                y_ = torch.cat([discriminator(imgs), discriminator(rec)], 0)[:, 0]
                y = torch.cat([torch.ones(len(imgs)), torch.zeros(len(rec))], 0).to(device)

                loss_g = loss - 0.2 * loss_fn(y_, y)

                opt.zero_grad()
                loss_g.backward()
                opt.step()

                rec_ = rec.detach()
                y_ = torch.cat([discriminator(imgs), discriminator(rec_)], 0)[:, 0]
                y = torch.cat([torch.ones(len(imgs)), torch.zeros(len(rec_))], 0).to(device)

                if batch_idx == 200:
                    print(y_)

                loss_d = loss_fn(y_, y)

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            
                if batch_idx % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {3:.2f} ({4:.2f})\tLoss-d  {5:.2f}'.format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                        avg_loss / avg_loss_cnt,
                        loss_d.item()))

            epoch += 1

            if epoch % args.save_freq == 0:
                save_model(model, loss_fn, opt, epoch)

            for i in range(5):
                save_img(imgs[i], 'true_epoch_' + str(epoch) + '_' + str(i) + '.png', test=False)
                save_img(output[i], 'pred_epoch_' + str(epoch) + '_' + str(i) + '.png', test=False)
            
            model.eval()
            avg_loss = 0
            avg_loss_cnt = 0
            for batch_idx, (imgs, _, _) in enumerate(test_loader):
                imgs = imgs.to(device).float()
                output = model(imgs)
                loss = loss_fn(output, imgs)
                avg_loss += loss.item()
                avg_loss_cnt += 1

            print('Epoch: [{0}]\t'
                'Test Loss {1:.2f}\t'.format(
                    epoch,
                    avg_loss / avg_loss_cnt))
            model.train()


    def save_img(data, name, test=True):
        data = torch.sigmoid(8 * (data - 0.5))
        true_data = data[0].detach().cpu().numpy() * 255
        true = im.fromarray(np.uint8(true_data))
        if test:
            true.save(args.results_path + 'test/' + name)
        else:
            true.save(args.results_path + 'train/' + name)


    def test(model, data_loader, loss_fn, epoch=0):
        model.eval()
        avg_loss = 0 
        avg_loss_cnt = 0
        for batch_idx, (imgs, _, _) in enumerate(data_loader):
            imgs = imgs.to(device).float()
            output = model(imgs)
            loss = loss_fn(output, imgs)
            avg_loss += loss.item()
            avg_loss_cnt += 1

            if batch_idx == 0:
                idx = list(range(20))
                encoded = model.encode(imgs[idx])

                #y = linear(encoded, ibf=5, ease=sigmoid)
                y = catmullRom(encoded, ibf=20, ease=iden)
                #y = bspline(encoded, ibf=5, ease=iden)
                y = y.to(device)

                output = model.decode(y)

                for i in range(len(output)):
                    save_img(output[i], str(i) + '.png')
                
                return

        print('Epoch (Test): [{0}]\t'
            'Loss {1:.2f}\t'.format(
                            epoch,
                            avg_loss / avg_loss_cnt))


    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.results_path + 'train/', exist_ok=True)
    os.makedirs(args.results_path + 'test/', exist_ok=True)

    # class_list = os.listdir('data/png')
    class_list = ['airplane'] # , 'apple', 'bear', 'bicycle', 'bird', 'broccoli', 'The Eiffel Tower', 'The Mona Lisa']
    train_dataset = SketchDataset(args.data_path, train=True, class_list=class_list, first_k=15000)
    test_dataset = SketchDataset(args.data_path, train=False, class_list=class_list, first_k=3000)

    model = Network(args, input_size=(1, 256, 256))
    model.to(device) 
    
    discriminator = Discriminator(args, input_size=(1, 256, 256))
    discriminator.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

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

    model, loss_fn, opt, epoch = load_model(model, loss_fn, opt, 20)
    # train(model, discriminator, train_loader, test_loader, loss_fn, opt, opt_d, 0)
    test(model, test_loader, loss_fn, 0)


    # This part is for generating data.json

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=128,
    #     shuffle=True)

    # paths_li = list()
    # labels_li = list()
    # encoded_li = list()

    # for i in range(15):
    #     imgs, labels, paths = next(iter(test_loader))
    #     labels = labels.cpu().detach().numpy()
    #     encoded = model.encode(imgs.cuda()).cpu().detach().numpy()

    #     labels = [int(l) for l in labels]
    #     encoded = [[float(x) for x in enc] for enc in encoded]

    #     paths_li.extend(paths)
    #     labels_li.extend(labels)
    #     encoded_li.extend(encoded)

    # from shutil import copy
    # for path in paths_li:
    #     # images saved to "png2" folder to avoid file name conflicts
    #     # image paths still point to folder "png"
    #     p = path.replace('png', 'png2', 1)
    #     s = '/'.join(p.split('/')[:-1])
    #     os.makedirs(s, exist_ok=True)
    #     copy(path, p)

    # save2json(paths_li, labels_li, encoded_li)
