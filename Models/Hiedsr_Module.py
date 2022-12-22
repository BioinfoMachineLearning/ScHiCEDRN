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
from Models.Hiedsr import edsr
from Utils.loss import Hicarn_loss as Hiss


class Hiedsr_2D(pl.LightningModule):

    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            mode_type = 'U_Net',
            ):

        super(Hiedsr_2D, self).__init__()
        self.lr = 1e-5
        self.gamma = 0.99
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.loss = Hiss.GeneratorLoss()
        self.mode = mode_type
        self.mse_loss = nn.MSELoss()
        self.snr_lambda = 1e-3
        self.Net = edsr()

        # initialise weights
        self.Net.init_params()

    def forward(self, inputs):

        final = self.Net(inputs) # F.sigmoid(final)
        return final

    def total_loss(self, output, target):  # for total
        return self.loss(output, target)

    def training_step(self, batch, batch_idx):
        data, target, _               = batch
        output                    = self.forward(data)

        mse, perception, tv, loss = self.total_loss(output, target)

        self.log('carn_train_loss/mse', mse)
        self.log('carn_train_loss/perception', perception)
        self.log('carn_train_loss/tv', tv)
        self.log('carn_train_loss/total', loss)
        return loss

    def validation_step(self, batch, batch_idx):  # output_size = target_size = 65
        data, target, info  = batch
        output       = self.forward(data)
        # MSE_loss, _, _, _  = self.total_loss(output, target)
        MSE_loss = self.mse_loss(output, target)
        self.log('carn_validation_loss', MSE_loss)
        return MSE_loss

    def test_step(self, batch, batch_idx):  # output_size = target_size = 65
        data, target, info  = batch
        output       = self.forward(data)
        # MSE_loss, _, _, _  = self.total_loss(output, target)
        MSE_loss = self.mse_loss(output, target)
        self.log('carn_test_loss', MSE_loss)
        return MSE_loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters( ),
                                     lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma = self.gamma)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        return [optimizer], [scheduler]
