##########################################################
# Code adapted from https://github.com/JiaRenChang/PSMNet
# Pyramid Stereo Matching Network,
# Chang, Jia-Ren and Chen, Yong-Sheng,
# CVPR 2018
##########################################################

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, use_bn=True):

    if use_bn:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                             nn.BatchNorm2d(out_planes))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, use_bn=True):

    if use_bn:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                             nn.BatchNorm3d(out_planes))
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,
                         bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, use_bn=True):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation, use_bn=use_bn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, use_bn=use_bn)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class feature_extraction(nn.Module):
    def __init__(self, no_SPP=False, use_bn=True, big_SPP=False):
        super(feature_extraction, self).__init__()

        self.no_SPP = no_SPP

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1, use_bn=use_bn),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, use_bn=use_bn),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, use_bn=use_bn),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1, use_bn=use_bn)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1, use_bn=use_bn)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1, use_bn=use_bn)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2, use_bn=use_bn)

        self.big_SPP = big_SPP

        if not self.no_SPP:

            if self.big_SPP:
                self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                             convbn(128, 32, 1, 1, 0, 1),
                                             nn.ReLU(inplace=True))

                self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                             convbn(128, 32, 1, 1, 0, 1),
                                             nn.ReLU(inplace=True))

                self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                             convbn(128, 32, 1, 1, 0, 1),
                                             nn.ReLU(inplace=True))

                self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                             convbn(128, 32, 1, 1, 0, 1),
                                             nn.ReLU(inplace=True))

            else:
                # low resolution (avg pool 64 is too big for input, throws an error)
                self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(64,64)),
                                             convbn(128, 32, 1, 1, 0, 1, use_bn=use_bn),
                                             nn.ReLU(inplace=True))

                self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(32,32)),
                                             convbn(128, 32, 1, 1, 0, 1, use_bn=use_bn),
                                             nn.ReLU(inplace=True))

                self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(16,16)),
                                             convbn(128, 32, 1, 1, 0, 1, use_bn=use_bn),
                                             nn.ReLU(inplace=True))

                self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(8,8)),
                                             convbn(128, 32, 1, 1, 0, 1, use_bn=use_bn),
                                             nn.ReLU(inplace=True))

        lastconv_input = 320 if not self.no_SPP else 192
        self.lastconv = nn.Sequential(convbn(lastconv_input, 128, 3, 1, 1, 1, use_bn=use_bn),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation, use_bn=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
               downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),)
            else:
                downsample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                       stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation, use_bn=use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation,use_bn=use_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        if not self.no_SPP:
            output_branch1 = self.branch1(output_skip)
            output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

            output_branch2 = self.branch2(output_skip)
            output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

            output_branch3 = self.branch3(output_skip)
            output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

            output_branch4 = self.branch4(output_skip)
            output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

            output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        else:
            output_feature = torch.cat((output_raw, output_skip), 1)

        output_feature = self.lastconv(output_feature)

        return output_feature