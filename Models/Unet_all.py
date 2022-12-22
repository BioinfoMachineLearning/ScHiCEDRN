#The Unet-parts component
#with adjustments to architecture
#the stencil code comes from

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, is_batchnorm, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if is_batchnorm:
            self.double_conv = nn.Sequential(  # this every per Conv2d has Hout = Hin
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:   # if False, then the double_Conv exludes the BatchNorm2e()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            )
        self.init_params()

    def forward(self, x):
        out = self.double_conv(x)
        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')


class unetUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, is_deconv):
        super().__init__()

        if is_deconv:     # every ConvTranspose2d has Hout = 2 * Hin, and every DoubleConv has Hout = Hin
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        else:   # every Upsample has Hout = 2 * Hin, and every DoubleConv has Hout = Hin
            # self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)  # this equals to nn.upsamplingBilinear2d(scale_factor = 2)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            )

        self.init_params()

    def forward(self, x):
        x = self.up(x)
        return x

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')


class Res_block(nn.Module):
    def __init__(self, out_channels, t = 2):
        super().__init__()

        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
        x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, t = 2):
        super().__init__()

        self.RRCNN = nn.Sequential(
            Res_block(out_channels, t),
            Res_block(out_channels, t)
        )

        self.Conv_1X1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.init_params()

    def forward(self, x):
        x = self.Conv_1X1(x)
        x1 = self.RRCNN(x)
        return x + x1

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
        )

        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace = True)
        self.init_params()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')

class U_Net(nn.Module):
    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            t = 2,
            ):
        super().__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.t = t

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        #downsampling
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_conv4 = unetConv2(filters[4], filters[3], self.is_batchnorm)

        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_conv3 = unetConv2(filters[3], filters[2], self.is_batchnorm)

        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_conv2 = unetConv2(filters[2], filters[1], self.is_batchnorm)

        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_conv1 = unetConv2(filters[1], filters[0], self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)

        # initialise weights
        self.init_params()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        center = self.center(maxpool4)

        de4 = self.up4(center)
        de4 = torch.cat((conv4, de4), dim = 1)
        de4 = self.up_conv4(de4)

        de3 = self.up3(de4)
        de3 = torch.cat((conv3, de3), dim = 1)
        de3 = self.up_conv3(de3)

        de2 = self.up2(de3)
        de2 = torch.cat((conv2, de2), dim = 1)
        de2 = self.up_conv2(de2)

        de1 = self.up1(de2)
        de1 = torch.cat((conv1, de1), dim = 1)
        de1 = self.up_conv1(de1)

        final = self.final(de1)
        return final

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')


class R2U_Net(nn.Module):
    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            t = 2,
            ):
        super().__init__()

        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.t = t

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = RRCNN_block(in_channels = self.in_channels, out_channels = filters[0], t = self.t)
        self.conv2 = RRCNN_block(in_channels = filters[0], out_channels = filters[1], t = self.t)
        self.conv3 = RRCNN_block(in_channels = filters[1], out_channels = filters[2], t = self.t)
        self.conv4 = RRCNN_block(in_channels = filters[2], out_channels = filters[3], t = self.t)

        self.center = RRCNN_block(in_channels = filters[3], out_channels = filters[4], t = self.t)

        #upsampling
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_conv4 = RRCNN_block(in_channels = filters[4], out_channels = filters[3], t = self.t)

        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_conv3 = RRCNN_block(in_channels = filters[3], out_channels = filters[2], t = self.t)

        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_conv2 = RRCNN_block(in_channels = filters[2], out_channels = filters[1], t = self.t)

        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_conv1 = RRCNN_block(in_channels = filters[1], out_channels = filters[0], t = self.t)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)

        # initialise weights
        self.init_params()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        center = self.center(maxpool4)

        de4 = self.up4(center)
        de4 = torch.cat((conv4, de4), dim = 1)
        de4 = self.up_conv4(de4)

        de3 = self.up3(de4)
        de3 = torch.cat((conv3, de3), dim = 1)
        de3 = self.up_conv3(de3)

        de2 = self.up2(de3)
        de2 = torch.cat((conv2, de2), dim = 1)
        de2 = self.up_conv2(de2)

        de1 = self.up1(de2)
        de1 = torch.cat((conv1, de1), dim = 1)
        de1 = self.up_conv1(de1)

        final = self.final(de1)
        return final

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')



class AttU_Net(nn.Module):
    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            t = 2
            ):
        super().__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.t = t

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        #downsampling
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.att4 = Attention_block(filters[3], filters[3], int(filters[3]/2))
        self.up_conv4 = unetConv2(filters[4], filters[3], self.is_batchnorm)

        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.att3 = Attention_block(filters[2], filters[2], int(filters[2]/2))
        self.up_conv3 = unetConv2(filters[3], filters[2], self.is_batchnorm)

        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.att2 = Attention_block(filters[1], filters[1], int(filters[1]/2))
        self.up_conv2 = unetConv2(filters[2], filters[1], self.is_batchnorm)

        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.att1 = Attention_block(filters[0], filters[0], int(filters[0]/2))
        self.up_conv1 = unetConv2(filters[1], filters[0], self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)

        # initialise weights
        self.init_params()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        center = self.center(maxpool4)

        de4 = self.up4(center)
        gat4 = self.att4(g = de4, x = conv4)
        de4 = torch.cat((gat4, de4), dim = 1)
        de4 = self.up_conv4(de4)

        de3 = self.up3(de4)
        gat3 = self.att3(g = de3, x = conv3)
        de3 = torch.cat((gat3, de3), dim = 1)
        de3 = self.up_conv3(de3)

        de2 = self.up2(de3)
        gat2 = self.att2(g = de2, x = conv2)
        de2 = torch.cat((gat2, de2), dim = 1)
        de2 = self.up_conv2(de2)

        de1 = self.up1(de2)
        gat1 = self.att1(g = de1, x = conv1)
        de1 = torch.cat((gat1, de1), dim = 1)
        de1 = self.up_conv1(de1)

        final = self.final(de1)
        return final

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')


class R2AttU_Net(nn.Module):
    def __init__(self,
            feature_scale = 4,
            n_classes = 1,
            is_deconv = True,
            in_channels = 1,
            is_batchnorm = True,
            t = 2,
            ):
        super().__init__()

        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.t = t

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = RRCNN_block(in_channels = self.in_channels, out_channels = filters[0], t = self.t)
        self.conv2 = RRCNN_block(in_channels = filters[0], out_channels = filters[1], t = self.t)
        self.conv3 = RRCNN_block(in_channels = filters[1], out_channels = filters[2], t = self.t)
        self.conv4 = RRCNN_block(in_channels = filters[2], out_channels = filters[3], t = self.t)

        self.center = RRCNN_block(in_channels = filters[3], out_channels = filters[4], t = self.t)

        #upsampling
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.att4 = Attention_block(filters[3], filters[3], int(filters[3] / 2))
        self.up_conv4 = RRCNN_block(in_channels = filters[4], out_channels = filters[3], t = self.t)

        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.att3 = Attention_block(filters[2], filters[2], int(filters[2] / 2))
        self.up_conv3 = RRCNN_block(in_channels = filters[3], out_channels = filters[2], t = self.t)

        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.att2 = Attention_block(filters[1], filters[1], int(filters[1] / 2))
        self.up_conv2 = RRCNN_block(in_channels = filters[2], out_channels = filters[1], t = self.t)

        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.att1 = Attention_block(filters[0], filters[0], int(filters[0] / 2))
        self.up_conv1 = RRCNN_block(in_channels = filters[1], out_channels = filters[0], t = self.t)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)

        # initialise weights
        self.init_params()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        center = self.center(maxpool4)

        de4 = self.up4(center)
        gat4 = self.att4(g = de4, x = conv4)
        de4 = torch.cat((gat4, de4), dim = 1)
        de4 = self.up_conv4(de4)

        de3 = self.up3(de4)
        gat3 = self.att3(g = de3, x = conv3)
        de3 = torch.cat((gat3, de3), dim = 1)
        de3 = self.up_conv3(de3)

        de2 = self.up2(de3)
        gat2 = self.att2(g = de2, x = conv2)
        de2 = torch.cat((gat2, de2), dim = 1)
        de2 = self.up_conv2(de2)

        de1 = self.up1(de2)
        gat1 = self.att1(g = de1, x = conv1)
        de1 = torch.cat((gat1, de1), dim = 1)
        de1 = self.up_conv1(de1)

        final = self.final(de1)
        return final

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')

