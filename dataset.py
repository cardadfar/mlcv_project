from torch.utils.data import Dataset
import numpy as np


class NPYDataset(Dataset):
    def __init__(self, data_path, train=True):
        super().__init__()
        data = np.load(data_path)
        n = len(data)
        split = int(0.9 * n)
        if train:
            self.data = data[:split]
            self.len = split
        else:
            self.data = data[split:]
            self.len = n - split

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index].reshape((28,28))