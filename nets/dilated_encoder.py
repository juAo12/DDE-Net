import torch.nn as nn
# from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm, normal_init)
from torch.nn import BatchNorm2d

# from ..builder import NECKS
from torchsummary import summary


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)
        return out


# @NECKS.register_module()
class DilateEncoder(nn.Module):
    # def __init__(self, in_channels, out_channels, lateral_channels, block_mid_channels, num_residual_blocks):
    def __init__(self, in_channels, out_channels, block_mid_channels, num_residual_blocks):
        super(DilateEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]


        self._init_layers()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.lateral_norm = BatchNorm2d(self.out_channels)
        self.fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_norm = BatchNorm2d(self.out_channels)
        self.encoder_blocks1 = Bottleneck(self.out_channels, self.block_mid_channels, dilation=self.block_dilations[0])
        self.encoder_blocks2 = Bottleneck(self.out_channels, self.block_mid_channels, dilation=self.block_dilations[1])
        self.encoder_blocks3 = Bottleneck(self.out_channels, self.block_mid_channels, dilation=self.block_dilations[2])
        self.encoder_blocks4 = Bottleneck(self.out_channels, self.block_mid_channels, dilation=self.block_dilations[3])

    def forward(self, x):
        out = self.lateral_norm(self.lateral_conv(x))
        out = self.fpn_norm(self.fpn_conv(out))
        out1 = out + self.encoder_blocks1(out)
        out2 = out + out1 + self.encoder_blocks2(out1)
        out3 = out + out1 + out2 + self.encoder_blocks3(out2)
        out4 = out + out1 + out2 + out3 + self.encoder_blocks4(out3)
        return out4
