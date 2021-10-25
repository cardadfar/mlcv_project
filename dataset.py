from torch.utils.data import Dataset
import numpy as np
import os

class NPYDataset(Dataset):
    def __init__(self, data_path, train=True, classList=None):
        super().__init__()
        
        first = True
        nFiles = len(os.listdir(data_path))
        data_list = []
        for idx, filename in enumerate(os.listdir(data_path)):

            dataset = filename[:-4]  # remove .npy
            if classList != None and dataset not in classList:
                continue 

            if train:
                print('Loading Train Data: [{0}/{1}] \t {2}'.format(idx, 
                                                        nFiles,
                                                        data_path + filename))
            else:
                print('Loading Test Data: [{0}/{1}] \t {2}'.format(idx, 
                                                        nFiles,
                                                        data_path + filename))

            data_raw = np.load(data_path + filename)

            data_raw_size = len(data_raw)
            split = int(0.9 * data_raw_size)

            if train:
                data = data_raw[:split]
                data_size = split
            else:
                data = data_raw[split:]
                data_size = data_raw_size - split

            data_list.append(data)
        
        self.data = np.concatenate(data_list, axis=0)
        self.len = len(self.data)
        
        print('Loades {0} Images From {1} Classes.'.format(self.len, nFiles))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index].reshape((28,28))