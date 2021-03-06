import pathlib
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

class SketchDataset(Dataset):
    def __init__(self, data_dir, train=True, class_list=None, first_k=2000):
        super().__init__()
        
        n_classes = len(class_list)

        if class_list is None:
            class_list = os.listdir(data_dir)

        self.img_paths = []
        self.labels = []

        self.class_list = class_list

        for k, classname in enumerate(class_list):
            img_dir = os.path.join(data_dir, classname)
            n_imgs = len(os.listdir(img_dir))

            split = int(n_imgs * 0.9)
            order = np.random.permutation(n_imgs)

            if train:
                for i in order[:first_k]:
                    img_path = os.path.join(img_dir, str(i) + '.png').replace('\\', '/')
                    self.img_paths.append(img_path)
                    self.labels.append(k)
            else:
                for i in order[split:split + first_k]:
                    img_path = os.path.join(img_dir, str(i) + '.png').replace('\\', '/')
                    self.img_paths.append(img_path)
                    self.labels.append(k)

        self.len = len(self.img_paths)
        print('Loaded {0} images paths from {1} classes.'.format(self.len, n_classes))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert('L')
        img = T.ToTensor()(img)
        label = self.labels[index]
        return img, label, path



class TestDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        

        self.class_list = os.listdir(data_dir)

        self.img_paths = []
        self.labels = []

        for k, classname in enumerate(self.class_list):
            img_dir = os.path.join(data_dir, classname)
            self.img_paths.append(img_dir)
            self.labels.append(-1)

        self.len = len(self.img_paths)
        print('Loaded {0} images.'.format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert('L')
        img = T.ToTensor()(img)
        label = self.labels[index]
        return img, label, path