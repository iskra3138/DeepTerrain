import numpy as np
import os
import torch
import gin
import scipy.io as sio
import torch.utils.data as data

@gin.configurable
class NBYJDataset(data.Dataset):
    """A data loader for NB dataset
    """
    def __init__(self, split, dataset_name, dataroot, size=512):
        self.split = split  # train, val, test
        self.dataset_name = dataset_name
        #self.input_list = input_list
        self.input_nc = 30
        self.size = size
        self.get_paths(split, dataset_name, dataroot)

    def get_paths(self, split, dataset_name, dataroot):
        #sub_set = 'Top'
        self.dataset_path = os.path.join(dataroot, 'NB_abs_knn') #, sub_set, split, 'inputs')  # path for inputs
        #self.B_path = os.path.join(dataroot, self.dataset_name, sub_set, split, 'dtm')  # path for DTMs

        self.split_list_file = os.path.join(dataroot, 'splits', '{}-{}.tiles'.format('NB', split))
        self.filenames = []
        with open(self.split_list_file, 'r') as f:
            lines = f.readlines()

        for line in lines :
            str1, str2 = line.strip().split('_')
            npyfilename = 'nb_2018_{}00_{}00.npy'.format(str1, str2)
            self.filenames.append(npyfilename)

        self.filenames = sorted(self.filenames)  # list of filename

    def scaling (self, A, B, minz, maxz) :
        """
        array3D: [ch, W, H]
        """
        # Scale B into [-1, 1]
        B = (2 * (B - minz) / (maxz - minz)) - 1

        A[2::3, :, :] = (2 * (A[2::3, :, :]  - minz) / (maxz - minz)) - 1
        #A[::3, :, :] = (A[::3, :, :] - 0.5) / 499.5
        #A[1::3, :, :] = (A[1::3, :, :]  - 0.5) / 499.5
        #for i in range(A.shape[0]):
        #    if input_list[i] in ['voxel-top', 'voxel-bottom', 'pixel-mean'] :
        #        A[i, :, :] = (2 * (A[i, :, :] - minz) / (maxz - minz)) - 1
        #    else:
        #        ch_min = A[i, :, :].min()
        #        ch_max = A[i, :, :].max()
        #        A[i, :, :] = (A[i, :, :] - ch_min) / (ch_max - ch_min)

        return A, B

    def padding (self, arr, left, right, top, bot) :
        """
        args:
            arr: [ch, h, w]
        return
            arr: [ch, 512, 512]
        """
        for i in range(left):
            arr = np.concatenate((arr[:, :1, :], arr), axis=1)
        for i in range(right):
            arr = np.concatenate((arr, arr[:, -1:, :]), axis=1)
        for j in range(top):
            arr = np.concatenate((arr[:, :, :1], arr), axis=2)
        for j in range(bot):
            arr = np.concatenate((arr, arr[:, :, -1:]), axis=2)

        return arr

    def random_crop(self, A, B):
        """
        args:
            array3D: [ch, 512, 512]
        return
            array3D: [ch, 256, 256]
        """
        x, y = np.random.randint(256, size=2)
        A = A[:, x:x + 256, y:y + 256]
        B = B[:, x:x + 256, y:y + 256]

        return A, B

    def random_flip(self, A, B):
        """
        args:
            array3D: [ch, 256, 256]
        return
            array3D: [ch, 256, 256]
        """
        h, v = np.random.rand(2)
        if h > 0.5 :
            A = np.flip(A, axis=1)
            B = np.flip(B, axis=1)
        if v > 0.5:
            A = np.flip(A, axis=2)
            B = np.flip(B, axis=2)

        return A, B

    def __getitem__(self, index):
        npyfile = os.path.join(self.dataset_path, self.filenames[index])
        mat = sio.loadmat(os.path.join('/nas2/YJ/DATA/ALS2DTM/NB/', self.filenames[index].replace('npy', 'mat')))
        A = np.load(npyfile)
        A = np.moveaxis(A, -1, 0)

        # Load B
        B = np.expand_dims(mat['dtm'], axis=0)

        # Make A
        #A = []
        #for key in self.input_list :
        #    input_raster = np.expand_dims(mat[key], axis=0)
        #    A.append(input_raster)
        #A = np.concatenate(A, axis=0)

        # Load semantics
        seg = np.expand_dims(mat['semantics'], axis=0)

        # Scaling
        ### save min_z, max_z to scale B into [-1, 1] since output of net is tanh
        min_z = B.min()
        max_z = B.max()
        ### For elevation rasters, all values will be scaled with min_z and max_z
        ### For statistic rasters, each raster will be scaled to [0, 1]
        A, B = self.scaling(A, B, min_z, max_z)

        # Padding
        diff_h, diff_w = self.size - B.shape[1], self.size - B.shape[2]
        left = diff_h // 2
        right = diff_h - left
        top = diff_w // 2
        bot = diff_w - top

        A = self.padding(A, 6, 6, 6, 6)
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
                    'B_min': min_z,
                    'B_max': max_z,
                    'filename': self.filenames[index]
                    }
        else :
            return {'A': A,
                    'B': B,
                    'seg': seg,
                    'B_min': min_z,
                    'B_max': max_z,
                    'filename': self.filenames[index],
                    'shape': B.shape[1:3]
                    }

    def __len__(self):
        return len(self.filenames)
