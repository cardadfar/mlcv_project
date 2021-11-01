import pathlib
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

class SketchDataset(Dataset):
    def __init__(self, data_dir, train=True, class_list=None):
        super().__init__()
        
        n_classes = len(class_list)
        data_list = []

        if class_list is None:
            class_list = os.listdir(data_dir)

        self.img_paths = []

        for classname in class_list:
            img_dir = os.path.join(data_dir, classname)
            n_imgs = len(os.listdir(img_dir))

            split = int(n_imgs * 0.9)

            if train:
                for i in range(split):
                    img_path = os.path.join(img_dir, str(i) + '.png')
                    self.img_paths.append(img_path)
            else:
                for i in range(split, n_imgs):
                    img_path = os.path.join(img_dir, str(i) + '.png')
                    self.img_paths.append(img_path)

        self.len = len(self.img_paths)
        print('Loaded {0} images paths from {1} classes.'.format(self.len, n_classes))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('L')
        img = T.ToTensor()(img)
        return img
