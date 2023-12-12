import numpy as np
import os
import torch
import gin
import scipy.io as sio
from dataloaders.dataset import DatasetTemplate

@gin.configurable
class SUIDataset(DatasetTemplate):
    """A data loader for NB dataset
    """
    def __init__(self, split, dataset_name, input_list, dataroot, pred_type='tanh', size=512):
        super().__init__(split, dataset_name, input_list, dataroot, pred_type=pred_type, size=size)
        self.get_paths(split, dataset_name, dataroot)

    def get_paths(self, split, dataset_name, dataroot):
        self.dataset_path = os.path.join(dataroot, self.dataset_name) #, sub_set, split, 'inputs')  # path for inputs

        self.filenames = [f for f in os.listdir(os.path.join(self.dataset_path, 'dtm')) if 'mat' in f]
        self.filenames = sorted(self.filenames)  # list of filename


    def __getitem__(self, index):
        dtmfile = os.path.join(self.dataset_path, 'dtm', self.filenames[index])
        dtm = sio.loadmat(dtmfile)['dtm']
        # Load B
        B = np.expand_dims(dtm, axis=0)

        # Load A
        dsmfile = os.path.join(self.dataset_path, 'dsm', self.filenames[index])
        dsm = sio.loadmat(dsmfile)['dsm']
        A = np.expand_dims(dsm, axis=0)

        # Load semantics
        segfile = os.path.join(self.dataset_path, 'seg', self.filenames[index])
        seg = sio.loadmat(segfile)['seg']
        seg = np.expand_dims(seg, axis=0)

        # Scaling
        ### save min_z, max_z to scale B into [-1, 1] since output of net is tanh
        min_z = A.min() #
        #min_z = B.min()
        max_z = A.max() #B.max()
        ### For elevation rasters, all values will be scaled with min_z and max_z
        ### For statistic rasters, each raster will be scaled to [0, 1]
        if self.pred_type == 'tanh' :
            A, B = self.scaling_tanh(A, B, min_z, max_z)
        else :
            A, B = self.scaling_sigmoid(A, B, min_z, max_z)
        # Padding
        diff_h, diff_w = self.size - B.shape[1], self.size - B.shape[2]
        left = diff_h // 2
        right = diff_h - left
        top = diff_w // 2
        bot = diff_w - top

        if diff_h !=0 or diff_w !=0 :
            A = self.padding(A, left, right, top, bot)
            B = self.padding(B, left, right, top, bot)
            seg = self.padding(seg, left, right, top, bot)

        # Cropping & Flipping
        if self.split == 'train' :
            #A, B = self.random_crop (A, B)
            A, B = self.random_flip(A, B)

        A = torch.from_numpy(A.astype(np.float32))
        B = torch.from_numpy(B.astype(np.float32))

        if self.split == 'train' :
            return {'A': A,
                    'B': B,
                    'A_min': min_z,
                    'A_max': max_z,
                    'filename': self.filenames[index]
                    }
        else :
            return {'A': A,
                    'B': B,
                    'seg': seg,
                    'A_min': min_z,
                    'A_max': max_z,
                    'filename': self.filenames[index],
                    'shape': B.shape[1:3]
                    }

    def __len__(self):
        return len(self.filenames)
