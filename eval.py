import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import gin

from dataloaders import build_dataloader
from models.pix2pix_model import Pix2PixModel

@gin.configurable
def eval(
        args,
        batch_size,
        num_threads,
        serial_batches,) :

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=> using '{}' for computation.".format(device))
    args.device = device

    # Build dataloader
    print("=> creating data loaders ... ")
    eval_loader, input_nc = build_dataloader(args.split,
                                             args.dataset,
                                             batch_size=batch_size,
                                             num_workers=num_threads,
                                             shuffle=not serial_batches,
                                             pin_momory=True
                                             )

    ## build_model
    print("=> creating model... ", end='')
    model = Pix2PixModel(args, input_nc=input_nc)
    model.setup(args)

    model.eval()
    total_iter = 0
    for i, data in enumerate(eval_loader):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        model.get_RMSE()
        total_iter += 1

    print(model.RMSE, total_iter, model.RMSE / total_iter)

def main():
    parser = argparse.ArgumentParser(description='DSM-to-DTM')
    parser.add_argument('--config', type=str, default='./config/NB/eval.gin', help='path of configures')
    parser.add_argument('--split', type=str, default='test', help='val or test')
    parser.add_argument('--dataset', type=str, default='NB', help='name of dataset')
    parser.add_argument('--name', type=str, default='als2dtm_pix2pix_debug',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

    args = parser.parse_args()
    #assert args.dataset in ['NB', 'DALES', 'DALESDSP'] , 'NB or DALES'
    #assert args.split in ['val', 'test'], 'val or test'
    gin.parse_config_file(args.config)

    eval(args)

if __name__ == '__main__':
    main()
