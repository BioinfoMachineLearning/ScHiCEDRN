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
from Models.schicedrn_gan import Generator, Discriminator
from Utils.loss import Hicarngan_loss as Hiss


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
        self.G_lr = 1e-5
        self.D_lr = 1e-5
        self.gamma = 0.99
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.Gloss = Hiss.GeneratorLoss()
        self.Dloss = nn.BCELoss()
        self.mode = mode_type
        self.mse_loss = nn.MSELoss()
        self.snr_lambda = 1e-3
        self.generator = Generator()
        self.discriminator = Discriminator()
        # initialise weights
        self.generator.init_params()
        self.discriminator.init_params()

    def forward(self, inputs):

        final = self.generator(inputs)
        return final

    """def G_loss(self, output, target):  # for total without LAD loss
        return self.Gloss(output, target)"""

    def G_loss(self, out_labels, output, target):  # for total without LAD loss
        return self.Gloss(out_labels, output, target)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, target, _               = batch
        fake_image = self.forward(data)

        # Generator:
        if optimizer_idx == 0:
            self.generator.zero_grad()
            pred_fake = self.discriminator(fake_image)
            fake_mean = pred_fake.mean()
            mse, perception, tv, loss = self.G_loss(fake_mean, fake_image, target)

            self.log('carn_train_loss/mse', mse)
            self.log('carn_train_loss/perception', perception)
            self.log('carn_train_loss/tv', tv)
            self.log('carn_train_loss/total', loss)
            return loss

        # Discriminator:
        if optimizer_idx == 1:
            self.discriminator.zero_grad()
            # training on fake image
            pred_fake = self.discriminator(fake_image)
            label_fake = torch.ones_like(pred_fake)
            d_loss_fake = self.Dloss(pred_fake, label_fake)

            # training on real image
            pred_real = self.discriminator(target)
            label_real = torch.ones_like(pred_real)
            d_loss_real = self.Dloss(pred_real, label_real)
            total_loss = d_loss_fake + d_loss_real

            self.log('carn_train_dloss/loss', total_loss)
            return total_loss

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
        opt_g = torch.optim.Adam(self.generator.parameters(), lr = self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr = self.D_lr)
        return [opt_g, opt_d]

