import torch
import torch.nn as nn
import torch.nn.functional as F


n_feat = 256
kernel_size = 3


class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        y = self.relu(self.res_conv(x))
        y = self.res_conv(y)
        y *= 0.1
        y = torch.add(y, x)
        return y


class edsr(nn.Module):  # 19M parameters
    def __init__(self):
        super(edsr, self).__init__()

        in_ch = 1
        number_blocks = 32
        self.head = nn.Conv2d(in_ch, n_feat, kernel_size, padding = 1)

        self.body = self.make_layer(_Res_Block, number_blocks)

        self.tail = nn.Conv2d(n_feat, in_ch, kernel_size, padding=1)

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        res_block.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding = 1))

        return nn.Sequential(*res_block)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        out = self.tail(res)

        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
