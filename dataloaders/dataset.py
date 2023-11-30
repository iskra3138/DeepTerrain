import os
import os.path
import glob
import numpy as np
import torch.utils.data as data


class DatasetTemplate(data.Dataset):
    """A data loader template for the all dataset
    """
    def __init__(self, split, dataset_name, dataroot): # need to path
        #self.args = args
        self.split = split # train, val, test
        self.dataset_name = dataset_name
        #self.input_nc = args.input_nc
        #self.output_nc = args.output_nc
        #self.get_paths(split, dataroot)
        #if self.split == 'train' :
        #    self.transform = self.train_transform
        #else :
        #    self.transform = self.no_transform

    def get_paths(self, split, dataroot):
        raise NotImplementedError

    def __getraw__(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.filenames)


