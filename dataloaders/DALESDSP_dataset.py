import numpy as np
import os
import torch
import gin
from dataloaders.dataset import DatasetTemplate

@gin.configurable
class DALESDSPDataset(DatasetTemplate):
    """A data loader for NB dataset
    """
    def __init__(self, split, dataset_name, input_list, dataroot, pred_type='tanh', size=512):
        super().__init__(split, dataset_name, input_list, dataroot, pred_type=pred_type, size=size)
        self.get_filelist(split, dataset_name, dataroot)

    def get_filelist(self, split, dataset_name, dataroot):
        self.dataset_path = os.path.join(dataroot, dataset_name, split) # path for npy files
        self.filenames = [f for f in os.listdir(os.path.join(self.dataset_path, 'dtm')) if 'npy' in f]
        self.filenames = sorted(self.filenames)  # list of filename

    def __getitem__(self, index):
        #matfile = os.path.join(self.dataset_path, self.filenames[index])
        dtm = np.load(os.path.join(self.dataset_path, 'dtm', self.filenames[index]))

        # Load B
        B = np.expand_dims(dtm, axis=0)

        # Make A
        A = []
        for key in self.input_list :
            input_raster = np.load(os.path.join(self.dataset_path, key, self.filenames[index]))
            input_raster = np.expand_dims(input_raster, axis=0)
            A.append(input_raster)
        A = np.concatenate(A, axis=0)

        # Load semantics
        seg = np.load(os.path.join(self.dataset_path, 'semantics', self.filenames[index]))
        seg = np.expand_dims(seg, axis=0)

        # Scaling
        ### save min_z, max_z to scale B into [-1, 1] since output of net is tanh
        bottom = np.load(os.path.join(self.dataset_path, 'voxel-bottom', self.filenames[index]))
        min_z = bottom.min()
        max_z = bottom.max()
        ### For elevation rasters, all values will be scaled with min_z and max_z
        ### For statistic rasters, each raster will be scaled to [0, 1]
        if self.pred_type == 'tanh':
            A, B = self.scaling_tanh(A, B, min_z, max_z, self.input_list)
        else:
            A, B = self.scaling_sigmoid(A, B, min_z, max_z, self.input_list)

        # Padding
        diff_h, diff_w = self.size - B.shape[1], self.size - B.shape[2]
        left = diff_h // 2
        right = diff_h - left
        top = diff_w // 2
        bot = diff_w - top

        A = self.padding(A, left, right, top, bot)
        B = self.padding(B, left, right, top, bot)
        seg = self.padding(seg, left, right, top, bot)

        # Cropping & Flipping
        if self.split == 'train' :
            A, B = self.random_crop (A, B)
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