#The Unet-parts component
#with adjustments to architecture
#the stencil code comes from
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_2D.py

import pdb
import numpy as np
import pytorch_lightning as pl
import math
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Trainer
from Models.Unet_all import U_Net, R2U_Net, AttU_Net, R2AttU_Net, init_weights
from Utils.loss import insulation as ins


class unet_2D(pl.LightningModule):

    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            mode_type = 'U_Net',
            ):

        super(unet_2D, self).__init__()
        self.lr = 1e-5
        self.gamma = 0.99
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.cross = nn.CrossEntropyLoss()
        self.tad = ins.InsulationLoss()
        self.mode = mode_type
        self.snr_lambda = 1e-3

        modes = ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']

        assert self.mode in modes

        if self.mode == modes[0]:
            self.Net = U_Net(feature_scale = self.feature_scale, n_classes = self.n_classes, is_deconv = self.is_deconv,
                                   in_channels = self.in_channels, is_batchnorm = self.is_batchnorm)
        elif self.mode == modes[1]:
            self.Net = R2U_Net(feature_scale = self.feature_scale, n_classes = self.n_classes, is_deconv = self.is_deconv,
                                   in_channels = self.in_channels, is_batchnorm = self.is_batchnorm)
        elif self.mode == modes[2]:
            self.Net = AttU_Net(feature_scale = self.feature_scale, n_classes = self.n_classes, is_deconv = self.is_deconv,
                                   in_channels = self.in_channels, is_batchnorm = self.is_batchnorm)
        else:
            self.Net = R2AttU_Net(feature_scale = self.feature_scale, n_classes = self.n_classes, is_deconv = self.is_deconv,
                                   in_channels = self.in_channels, is_batchnorm = self.is_batchnorm)

        # initialise weights
        self.Net.init_params()

    def forward(self, inputs):

        final = self.Net(inputs) # F.sigmoid(final)
        return F.sigmoid(final)

    def PSNR(self, output, target):
        MSE = torch.square(output[0][0] - target[0][0]).mean()
        MAX = torch.max(target)
        loss = (20*torch.log10(MAX) - 10*torch.log10(MSE))
        return loss

    def jaccard_distance(self, output, target, smooth = 100):
        mask_output = torch.ne(output, 0)
        mask_target = torch.ne(target, 0)
        mask = torch.logical_and(mask_output, mask_target)
        output = torch.masked_select(output, mask)
        target = torch.masked_select(target, mask)
        inter = torch.sum(torch.minimum(output, target))
        union = torch.sum(torch.maximum(output, target))
        dice = 1 - (inter + smooth)/(union + smooth)
        return dice

    def tad_loss(self, target, output):  # for insulation
        return self.tad(target, output)

    def criterion(self, output, target):
        return self.cross(output, target)

    def meanSquaredError_loss(self, target, output):
        return F.mse_loss(target, output)

    def training_step(self, batch, batch_idx):
        data, target, _               = batch
        output                    = self.forward(data)
        # print(f'the input data shape is {data.size( )} and the output shape is {output.size( )} and target shape is {target.size( )}')
        # print(data)
        # print(target)
        # print(output)
        mse = self.meanSquaredError_loss(output, target)
        dice = self.jaccard_distance(output, target)
        TAD_loss = self.tad_loss(output, target)
        PSNR_loss = self.PSNR(output, target)
        loss = mse + dice + TAD_loss + self.snr_lambda * PSNR_loss

        # self.criterion(output, target)
        # print(loss)
        # input("press enter to continue ....")
        # self.dice_loss(F.softmax(output, dim=1).float(), F.one_hot(target, 1).permute(0, 3, 1, 2).float(),multiclass = False)

        self.log('unet_train_loss/mse', mse)
        self.log('unet_train_loss/dice', dice)
        self.log('unet_train_loss/TAD', TAD_loss)
        self.log('unet_train_loss/PSNR', PSNR_loss)
        self.log('unet_train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):  # output_size = target_size = 65
        data, target, info  = batch
        output       = self.forward(data)
        MSE_loss     = self.meanSquaredError_loss(output, target)
        self.log('unet_validation_loss', MSE_loss)
        return MSE_loss

    def test_step(self, batch, batch_idx):  # output_size = target_size = 65
        data, target, info  = batch
        output       = self.forward(data)
        MSE_loss     = self.meanSquaredError_loss(output, target)
        self.log('unet_test_loss', MSE_loss)
        return MSE_loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters( ),
                                     lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma = self.gamma)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        return [optimizer], [scheduler]
