import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ..builder import BACKBONES
from .resnet import ResNet

import gc
from scipy.ndimage import distance_transform_edt
@BACKBONES.register_module
class MyBackBone(ResNet):
    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True
                 ):
        super().__init__(
            depth,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3),
            style='pytorch',
            frozen_stages=-1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            dcn=None,
            stage_with_dcn=(False, False, False, False),
            gcb=None,
            stage_with_gcb=(False, False, False, False),
            gen_attention=None,
            stage_with_gen_attention=((), (), (), ()),
            with_cp=False,
            zero_init_residual=True)

        self.edgenet = UNet(1, 1)
        # self.conv_test = nn.Conv2d(64,1,kernel_size=3,padding=1,bias=False,stride=1)
        # self.bn_test = nn.BatchNorm2d(64)
        # self.softmax = nn.LogSoftmax(dim=0)
        # self.nlloss = nn.NLLLoss2d()
        self.sigmiod = nn.Sigmoid()


    def forward(self, x, img_ori):

        # x.shape torch.Size([2, 3, 800, 800])
        x_size=x.size()
        x = self.conv1(x)  # self.conv1 Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) x.shape torch.Size([2, 64, 400, 400])
        x = self.norm1(x)
        x = self.relu(x)
        # import ipdb;
        # ipdb.set_trace()

        x_test = torch.mean(x, dim=1, keepdim=True)

        x_test = (x_test.cpu().detach().numpy()*255).astype(np.uint8)
        fusion_canny = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
        for i in range(x_test.shape[0]):
            x_test[i] = cv2.medianBlur(x_test[i][0], 5)
            fusion_canny[i] = cv2.Canny(x_test[i].transpose((1, 2, 0)), 10, 100)
        fusion_canny = torch.from_numpy(fusion_canny).cuda().float()
        # import matplotlib.pyplot as plt
        # plt.imshow(x_test[0][0])
        # plt.show()
        # plt.imshow(x_test[1][0])
        # plt.show()
        # plt.imshow(fusion_canny[0][0])
        # plt.show()
        # plt.imshow(fusion_canny[1][0])
        # plt.show()
        x = self.maxpool(x)  # x.shape torch.Size([2, 64, 200, 200])
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        im_arr = img_ori.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype('uint8')
        gray = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        # import ipdb;ipdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.imshow(im_arr[0])
        # plt.show()
        for i in range(x_size[0]):
            gray[i] = cv2.cvtColor(im_arr[i], cv2.COLOR_BGR2GRAY)
            filter_gray = cv2.medianBlur(gray[i][0].astype('uint8'),3)

            canny[i] = cv2.Canny(filter_gray, 10, 100)
            # plt.imshow(canny[i][0])
            # plt.show()

        canny = torch.from_numpy(canny).cuda().float()

        edge, edge_feat = self.edgenet(canny, fusion_canny)


        return tuple(outs), edge, edge_feat

    def edge_loss(self, edge_loss_inputs):
        # import ipdb;ipdb.set_trace()
        # import matplotlib.pyplot as plt
        pre_egde = edge_loss_inputs[0].squeeze(0)
        # pre_egde = self.sigmiod(pre_egde)
        # pre_egde=pre_egde.cpu().detach().numpy()
        # plt.imshow(pre_egde[0])
        # plt.show()
        # softmax_foreground = self.softmax(pre_egde)
        pre_egde = self.sigmiod(pre_egde)
        log_p = pre_egde.cuda()
        target = torch.tensor(edge_loss_inputs[1].astype(np.float32)).cuda()
        target_t = target.clone()
        # print(target.dtype)

        # edge_loss = self.nlloss(softmax_foreground[0],edge_loss_inputs[1])
        # c, h, w = pre_egde.size()

        # log_p = pre_egde.flatten().contiguous().view(1, -1)

        # log_p = pre_egde.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

        # target_t = target.flatten().contiguous().view(1, -1)

        # target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        assert log_p.shape==target_t.shape
        # target_trans = target_t.clone()
        #
        # pos_index = (target_t ==1)
        # neg_index = (target_t ==0)
        #
        #
        # target_trans[pos_index] = 1
        # target_trans[neg_index] = 0
        #
        # pos_index = pos_index.data.cpu().numpy().astype(bool)
        # neg_index = neg_index.data.cpu().numpy().astype(bool)

        edge_weight = distance_transform_edt(1-edge_loss_inputs[1])
        # weight = torch.Tensor(log_p.size()).fill_(0)
        # weight = torch.Tensor(edge_weight.flatten())
        weight = torch.Tensor(edge_weight)
        weight = weight.numpy()
        # pos_num = pos_index.sum()
        # neg_num = neg_index.sum()
        # sum_num = pos_num + neg_num
        # weight[pos_index] = neg_num*1.0 / sum_num
        # weight[neg_index] = pos_num*1.0 / sum_num


        weight = torch.from_numpy(weight)
        weight = weight.cuda()

        # import ipdb;
        # ipdb.set_trace()
        # loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
        return loss






class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.fusion = Fusion(128, 128)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, fusion_canny):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        # import ipdb;
        # ipdb.set_trace()
        x2 = self.fusion(x2, fusion_canny)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_a = x5
        x = self.up1(x5, x4) #torch.Size([1, 512, 100, 100])
        x_b = x
        x = self.up2(x, x3)  #torch.Size([1, 256, 200, 200])
        x_c = x
        x = self.up3(x, x2)  #torch.Size([1, 128, 400, 400])
        x_d = x
        x = self.up4(x, x1)  #torch.Size([1, 64, 800, 800])
        logits = self.outc(x)
        edge_feat = [x_a, x_b, x_c, x_d, x]
        return logits, edge_feat


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 bias=False):

        super(Fusion, self).__init__()

        self.fusion = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1,1),
            nn.ReLU(),
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels = out_channels,
                        kernel_size = kernel_size, stride = stride,bias = bias)

    def forward(self, input_feature, fusion_canny):
        fusion_feature = self.fusion(torch.cat([input_feature,fusion_canny],dim=1))
        input_feature = (input_feature*(fusion_feature+1))
        gc.collect()
        torch.cuda.empty_cache()
        return self.conv(input_feature)
