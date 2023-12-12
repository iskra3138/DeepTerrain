import os.path
import gin

import torch
import numpy as np

from .base_model import BaseModel
from . import networks


@gin.configurable
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    '''
    @staticmethod
    def modify_commandline_argsions(parser, is_train=True):
        """Add new dataset-specific argsions, and rewrite default values for existing argsions.

        Parameters:
            parser          -- original argsion parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser
    '''

    def __init__(self, args,
                 save_dir, input_nc, output_nc, ngf, ndf, netG, netD, norm, no_dropout, init_type,
                 init_gain,  n_layers_D, gan_mode, verbose, isTrain, load_iter,epoch, logger,
                 keys=None,
                 label_dics=None,
                 continue_train=False,
                 lr=0.0002,
                 beta1=0.5,
                 lambda_L1=100,
                 lr_policy='linear',
                 lr_decay_iters=50,
                 pred_type='tanh',
                 aux_loss='na',
                 size=512
                 ):
        """Initialize the pix2pix class.

        Parameters:
            args (Arguments class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, args,
                           save_dir, input_nc, output_nc, ngf, ndf, netG, netD, norm, no_dropout, init_type,
                           init_gain, n_layers_D, gan_mode, verbose, isTrain, load_iter, epoch, logger,
                           continue_train=continue_train,
                           lr=lr,
                           beta1=beta1,
                           lambda_L1=lambda_L1,
                           lr_policy=lr_policy,
                           lr_decay_iters=lr_decay_iters,
                           size=size
                           )
        assert pred_type in ['tanh', 'sigmoid'], 'tanh or sigmoid should be given'
        self.pred_type = pred_type
        assert aux_loss in ['gan', 'na', 'tv1', 'tv2'], 'na, gan, tv1 or tv2 should be given'
        self.aux_loss = aux_loss
        self.keys = keys
        self.label_dics = label_dics

        self.model_names = ['G']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.aux_loss == 'na' :
            self.loss_names = ['G_L1']
        elif self.aux_loss == 'gan' :
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
            # when only 'gan' loss, saved model_names will be 'G' and 'D'
            if self.isTrain:
                self.model_names = ['G', 'D']
        elif self.aux_loss == 'tv1' :
            self.loss_names = ['G_L1', 'TV1']
        else :
            self.loss_names = ['G_L1', 'TV2']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        #if self.isTrain:
        #    self.model_names = ['G'] #, 'D']
        #else:  # during test time, only load G
        #    self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(input_nc, output_nc, ngf, netG, norm, ## args = {3, 3, 64, unet_256, batch, True, normal, 0.02, 0}
                                      not no_dropout, init_type, init_gain, self.gpu_ids)

        if self.isTrain and self.aux_loss == 'gan' :  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(input_nc + output_nc, ndf, netD, ## args = {3, 3, 64, basic, 3, batch, normal, 0.02, 0}
                                          n_layers_D, norm, init_type, init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.aux_loss == 'gan' :
                self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        #AtoB = True # self.args.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.filename = input['filename']
        self.real_A_min = input['A_min'].to(self.device)
        self.real_A_max = input['A_max'].to(self.device)
        if not self.isTrain :
            self.seg = input['seg']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def back_to_org_scale (self, tensor, min, max) :
        if self.pred_type == 'tanh' :
            return min + (0.5 * (tensor + 1) * (max - min))
        else :
            return min + (tensor * (max - min))

    def save_prediction(self, preds):
        ## Save fake_B
        preds_save_dir = os.path.join(self.save_dir, 'preds_{}'.format(self.args.dataset))
        #self.logger.info('Results will be saved on %s' % preds_save_dir)
        if not os.path.exists(preds_save_dir):
            os.makedirs(preds_save_dir)
        for i, pred in enumerate(preds) :
            if 'mat' in self.filename[i] :
                filename = self.filename[i].replace('mat', 'npy')
            else :
                filename = self.filename[i]
            with open(os.path.join(preds_save_dir, filename), 'wb') as f:
                np.save(f, pred.cpu().numpy()[0])

    def get_classwise_RMSE(self, keys, seg, pred, gt):
        label_dics = self.label_dics

        seg = seg.cpu().numpy()
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        total_num = gt.shape[1] * gt.shape[2]

        others_index = np.ones_like(seg)
        rmse_dict = {}
        occup_dict = {}

        RMSE = np.sqrt(np.mean((pred - gt) ** 2))
        rmse_dict['total'] = np.round(RMSE, 2)

        # class-wise results
        for key in keys:
            # if key == 'ground' :
            #    mask = np.where((seg == 2) | (seg==8))
            #    num = mask[0].shape[0]
            #    others_index[mask] = 0
            if key == 'others':
                mask = np.where(others_index == 1)
                num = mask[0].shape[0]
            else:
                label = label_dics[key]
                mask = np.where(seg == label)
                num = mask[0].shape[0]
                others_index[mask] = 0
            if not num == 0:
                RMSE_cls = np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2))
            else:
                RMSE_cls = 0

            rmse_dict[key] = np.round(RMSE_cls, 2)
            occup_dict[key] = np.round(100 * num / total_num, 2)

        return RMSE, rmse_dict, occup_dict
    def get_RMSE(self):
        #self.RMSE += np.sqrt(((self.fake_B - self.real_A) ** 2).mean())
        # Denormalization
        self.fake_B = self.back_to_org_scale(self.fake_B, self.real_A_min, self.real_A_max) #(self.fake_B * self.real_B_max) + self.real_B_min
        self.real_B = self.back_to_org_scale(self.real_B, self.real_A_min, self.real_A_max) #(self.real_B * self.real_B_max) + self.real_B_min

        if not self.isTrain:
            self.save_prediction(self.fake_B)
            if self.size == 512 :
                dtm = self.real_B[:, :, 6:-6, 6:-6]
                pred = self.fake_B[:, :, 6:-6, 6:-6]
                seg = self.seg[:, :, 6:-6, 6:-6]
            else :
                dtm = self.real_B
                pred = self.fake_B
                seg = self.seg
            self.RMSE += torch.sqrt(torch.mean((pred - dtm) ** 2))

            for i in range(dtm.shape[0]) :
                RMSE, rmse_dict, occup_dict = self.get_classwise_RMSE(self.keys, seg[i], pred[i], dtm[i])
                self.RMSE2 += RMSE

                msg = ',' + self.filename[i] + ',' + str(rmse_dict['total'])
                for key in self.keys:
                    msg += ',' + str(rmse_dict[key])
                for key in self.keys:
                    msg += ',' + str(occup_dict[key])
                #msg += '\n'
                self.logger.info(msg)

        else :
            self.RMSE += torch.sqrt(torch.mean((self.fake_B - self.real_B) ** 2))
        return

    def total_variation_loss1(self, raster):
        tv_loss = torch.mean(torch.abs(raster[:, :, :, :-1] - raster[:, :, :, 1:])) + \
                  torch.mean(torch.abs(raster[:, :, :-1, :] - raster[:, :, 1:, :]))
        return tv_loss

    def total_variation_loss2(self, raster):
        tv_loss = torch.mean((raster[:, :, :, :-1] - raster[:, :, :, 1:]) ** 2) + \
                   torch.mean((raster[:, :, :-1, :] - raster[:, :, 1:, :]) ** 2)
        return tv_loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) # nn.BCEWithLogitsLoss()
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.aux_loss == 'gan' :
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        # combine loss and calculate gradients
        if self.aux_loss == 'gan' :
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        elif self.aux_loss == 'tv1' :
            self.loss_TV1 = self.total_variation_loss1(self.fake_B)
            self.loss_G = self.loss_G_L1 + self.loss_TV1
        elif self.aux_loss == 'tv2' :
            self.loss_TV2 = self.total_variation_loss2(self.fake_B)
            self.loss_G = self.loss_G_L1 + self.loss_TV2
        else :
            self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.aux_loss == 'gan' :
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
