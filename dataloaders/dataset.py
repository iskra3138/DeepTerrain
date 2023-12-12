import numpy as np
import torch.utils.data as data

#@gin.configurable
class DatasetTemplate(data.Dataset):
    """A data loader for NB dataset
    """
    def __init__(self, split, dataset_name, input_list, dataroot, pred_type='tanh', size=512):
        self.split = split  # train, val, test
        assert pred_type in ['tanh', 'sigmoid'], 'tanh or sigmoid should be given'
        self.pred_type = pred_type
        self.dataset_name = dataset_name
        self.input_list = input_list
        self.input_nc = len(input_list)
        self.size = size

    def get_paths(self, split, dataset_name, dataroot):
        raise NotImplementedError

    def scaling_tanh (self, A, B, minz, maxz, input_list=None) :
        """
        array3D: [ch, W, H]
        """
        # Scale B into around [-1, 1] with minz and maxz of voxel-bottom
        B = (2 * (B - minz) / (maxz - minz)) - 1
        if input_list is None :
            A = (2 * (A - minz) / (maxz - minz)) - 1
        else :
            for i in range(A.shape[0]):
                if input_list[i] in ['voxel-top', 'voxel-bottom', 'pixel-mean'] :
                    A[i, :, :] = (2 * (A[i, :, :] - minz) / (maxz - minz)) - 1
                else :
                    ch_min = A[i, :, :].min()
                    ch_max = A[i, :, :].max()
                    A[i, :, :] = (A[i, :, :] - ch_min) / (ch_max - ch_min)

        return A, B

    def scaling_sigmoid (self, A, B, minz, maxz, input_list=None) :
        """
        array3D: [ch, W, H]
        """
        # Scale B into around [0, 1] with minz and maxz of voxel-bottom
        B = (B - minz) / (maxz - minz)

        if input_list in None :
            A = (A - minz) / (maxz - minz)
        else :
            for i in range(A.shape[0]):
                if input_list[i] in ['voxel-top', 'voxel-bottom', 'pixel-mean'] :
                    A[i, :, :] = (A[i, :, :] - minz) / (maxz - minz)
                else :
                    ch_min = A[i, :, :].min()
                    ch_max = A[i, :, :].max()
                    A[i, :, :] = (A[i, :, :] - ch_min) / (ch_max - ch_min)

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

    def random_crop(self, A, B, start_point = 256):
        """
        args:
            array3D: [ch, 256+start_point, 256+start_point]
        return
            array3D: [ch, 256, 256]
        """
        x, y = np.random.randint(start_point, size=2)
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
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
