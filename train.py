import argparse
import os
import time
import gin
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from dataloaders import build_dataloader
from models.pix2pix_model import Pix2PixModel
from models.segformer import Segformer
from utils.utils import create_logger

def print_current_losses(epoch, iters, losses, t_comp, t_data, logger):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    #print(message)
    logger.info(message)


@gin.configurable
def train(
        args,
        batch_size,
        num_threads,
        serial_batches,
        epoch_count,
        n_epochs,
        n_epochs_decay,
        print_freq,
        save_latest_freq,
        save_epoch_freq,
        save_by_iter,
        save_dir,
        logger
) :
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
        torch.backends.cudnn.benchmark = True
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    ## build_dataloader
    #print("=> creating data loaders ... ")
    logger.info('------------------------- creating data loaders -------------------------')

    train_loader, input_nc = build_dataloader('train',
                                                   args.dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_threads,
                                                   shuffle=not serial_batches,
                                                   pin_momory=True,
                                                   logger=logger
                                                   )
    logger.info('------------------------- created data loaders -------------------------')
    logger.info('input ch: %d' % input_nc)
    #print (100*'#')
    #print ('input ch: ', input_nc)
    #print (100*'#')

    ## build_model
    #print("=> creating model and optimizer ... ", end='')
    logger.info('------------------------- creating model -------------------------')
    model = Pix2PixModel(args, save_dir=save_dir, input_nc=input_nc, logger=logger)
    model = Segformer()
    model.setup(epoch_count, n_epochs, n_epochs_decay)
    logger.info('------------------------- created model -------------------------')

    total_iters = 0

    # write config file
    #print(gin.operative_config_str())
    logger.info(gin.operative_config_str())
    with open(os.path.join(save_dir, "train_config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    # main loop
    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += batch_size
            epoch_iter += batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, logger)

            if total_iters % save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                #print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates after optimezer.step()
        """
        /root/anaconda3/envs/fpt/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:131: 
        UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, 
        you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. 
        See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`
        """

        if epoch % save_epoch_freq == 0:   # cache our model every <save_epoch_freq> epochs
            #print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            #model.save_networks(epoch)

        #print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
        logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
def main():
    parser = argparse.ArgumentParser(description='DSM-to-DTM')
    parser.add_argument('--config', type=str, default='./config/NB_Bottom/train.gin', help='path of configures')
    parser.add_argument('--dataset', type=str, default='NB', help='name of dataset')
    parser.add_argument('--name', type=str, default='als2dtm_pix2pix_debug',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--output_dir', type=str, default='./results', help='models are saved here')

    args = parser.parse_args()
    gin.parse_config_file(args.config)



    # Make save directory
    save_dir = os.path.join(args.output_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_name = os.path.join(save_dir, "train_log.txt")
    logger = create_logger(log_file_name)

    start = time.time()
    train(args, save_dir = save_dir, logger = logger)
    print ('total training time: ', time.time() - start)

if __name__ == '__main__':
    main()
