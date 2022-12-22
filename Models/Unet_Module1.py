#The Unet-parts component
#with adjustments to architecture
#the stencil code comes from
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_2D.py

import pdb
import pytorch_lightning as pl
import math
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Trainer
from Models.Unet_parts1 import unetConv2, unetUp, init_weights
from Utils.loss import insulation as ins


class unet_2D(pl.LightningModule):

    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            ):

        super(unet_2D, self).__init__()
        self.lr = 1e-5
        self.gamma = 0.99
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.cross = nn.CrossEntropyLoss()
        self.tad = ins.InsulationLoss()

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1) # F.sigmoid(final)
        return F.sigmoid(final)

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
        loss = mse + dice + TAD_loss

        # self.criterion(output, target)
        # print(loss)
        # input("press enter to continue ....")
        # self.dice_loss(F.softmax(output, dim=1).float(), F.one_hot(target, 1).permute(0, 3, 1, 2).float(),multiclass = False)

        self.log('unet_train_loss/mse', mse)
        self.log('unet_train_loss/dice', dice)
        self.log('unet_train_loss/TAD', TAD_loss)
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
