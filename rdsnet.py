from ..registry import DETECTORS
from .new_single_stage import MySingleStageDetector
from .. import builder
from mmdet.core import bbox2result, mask2result, multi_apply
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
#
from ..utils import ConvModule
from mmdet.core import auto_fp16
import numpy as np
from mmdet.ops import DeformConv
from ..builder import build_loss
from mmdet.core import (PointGenerator, multi_apply, multiclass_nms,
                        point_target)
from mmdet.models.anchor_heads.reppoints_head import RepPointsHead
#
@DETECTORS.register_module
class RDSNet(MySingleStageDetector):

    def __init__(self,
                 backbone,
                 bbox_head,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 mbrm_cfg=None,
                 fusion_cfg=None,
                 neck=None,
                 pretrained=None):
        super(RDSNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.mask_head = builder.build_head(mask_head)
        self.init_extra_weights()
        self.mbrm_cfg = mbrm_cfg
        if mbrm_cfg is not None:
            self.mbrm = MBRM(mbrm_cfg)
        else:
            self.mbrm = None

        # if fusion_cfg is not None:
        #     self.fusion=Fusion(fusion_cfg)
        # else:
        #     self.fusion=None

        self.trans = trans_QKV()

        self.attention1 = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=False)
        self.attention2 = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=False)
        self.attention3 = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=False)
        self.attention4 = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=False)
        self.attention5 = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=False)
        self.attention = nn.ModuleList([self.attention1, self.attention2, self.attention3, self.attention4, self.attention5])


        self.sigmiod = nn.Sigmoid()


    def init_extra_weights(self):
        self.mask_head.init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      img_ori,
                      edge,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        # import ipdb;
        # ipdb.set_trace()
        x, pre_edge, edge_feat = self.extract_feat(img, img_ori)
        #
        # img_tensor=torch.mean()
        edge_loss_inputs = list(pre_edge)+edge
        egde_losses = dict()
        egde_losses['edge_loss'] = self.backbone.edge_loss(edge_loss_inputs)



        x_b_size = [x[i].shape[2:] for i in range(len(x))]
        # x_size = []
        # batch = x[0].size(0)

        # for b in range(batch):
        #     for i in range(len(x)):
        #         x_size.append(x[i][b].shape[1:])
        #     x_b_size.append(x_size)
        # import ipdb;
        # ipdb.set_trace()
        edge_feat = self.trans(edge_feat, x_b_size)


        x = list(x)
        for i in range(5):
            # import ipdb;ipdb.set_trace()
            x[4-i] = self.attention[i](x[4-i], edge_feat[i])
        # x[4] = self.attention[0](x[4], edge_feat[0])

        x = tuple(x)
        del edge_feat
        gc.collect()
        torch.cuda.empty_cache()

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)

        losses, proposals = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # no positive proposals
        if proposals is None:
            return losses
        # needed in  mask generator
        c, h, w = img.size()[1:4]
        for img_meta in img_metas:
            img_meta['stack_shape'] = (h, w, c)
        # import ipdb;ipdb.set_trace()
        pred_masks = self.mask_head(x, proposals['pos_obj'], img_metas)


        losses_masks, final_masks, final_boxes = self.mask_head.loss(pred_masks, proposals['gt_bbox'],
                                                                     proposals['gt_mask'], img_metas, self.train_cfg)
        losses.update(losses_masks)
        # import ipdb;ipdb.set_trace()
        losses.update(egde_losses)

        return losses

    def simple_test(self, img, img_meta, img_ori, rescale=False):
        # import ipdb;
        # ipdb.set_trace()
        x, pre_edge, edge_feat = self.extract_feat(img, img_ori[0])
        # pre_edge =self.sigmiod(pre_edge)
        # import ipdb;ipdb.set_trace()

        # import matplotlib.pyplot as plt
        # pre= pre_edge.cpu().detach().numpy()
        # plt.imshow(pre[0][0])
        # plt.show()
        
        x_b_size = [x[i].shape[2:] for i in range(len(x))]

        edge_feat = self.trans(edge_feat, x_b_size)
        # # import ipdb;ipdb.set_trace()
        # edge_feat = self.trans(edge_feat)

        x = list(x)
        for i in range(5):
            # import ipdb;ipdb.set_trace()
            x[4-i] = self.attention[i](x[4-i], edge_feat[i])
        # x[4] = self.attention[0](x[4], edge_feat[0])

        x = tuple(x)
        del edge_feat
        gc.collect()
        torch.cuda.empty_cache()

        outs = self.bbox_head(x)

        # import ipdb;
        # ipdb.set_trace()
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bboxes = [bbox[0] for bbox in bbox_list]
        labels = [bbox[1] for bbox in bbox_list]
        pos_objs = [bbox[2] for bbox in bbox_list]
        # needed in mask generator
        c, h, w = img.size()[1:4]
        for meta in img_meta:
            meta['stack_shape'] = (h, w, c)
        # import ipdb;ipdb.set_trace()
        pred_masks = self.mask_head(x, pos_objs, img_meta)


        pred_masks = self.mask_head.get_masks(pred_masks, bboxes, img_meta, self.test_cfg, rescale=rescale)

        if self.mbrm is not None:
            bboxes = [self.mbrm.get_boxes(self.mbrm(pred_mask, bbox[:, :-1]), bbox[:, -1:])
                      for pred_mask, bbox in zip(pred_masks, bboxes)]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in zip(bboxes, labels)
        ]
        mask_results = [
            mask2result(pred_mask, det_labels, self.bbox_head.num_classes, self.test_cfg.mask_thr_binary)
            for pred_mask, det_labels in zip(pred_masks, labels)
        ]
        return bbox_results[0], mask_results[0]


class trans_QKV(nn.Module):
    def __init__(self, in_channels=[1024, 512, 256, 128, 64]):
        super(trans_QKV, self).__init__()

        self.module1 = nn.Conv2d(in_channels=in_channels[0], out_channels=256, kernel_size=1, stride=1, bias=False)
        self.module2 = nn.Conv2d(in_channels=in_channels[1], out_channels=256, kernel_size=1, stride=1, bias=False)
        self.module3 = nn.Conv2d(in_channels=in_channels[2], out_channels=256, kernel_size=1, stride=1, bias=False)
        self.module4 = nn.Conv2d(in_channels=in_channels[3], out_channels=256, kernel_size=1, stride=1, bias=False)
        self.module5 = nn.Conv2d(in_channels=in_channels[4], out_channels=256, kernel_size=1, stride=1, bias=False)

        self.module = nn.ModuleList([self.module1, self.module2, self.module3, self.module4, self.module5])
        # self.module = nn.ModuleList([self.module1, self.module2, self.module3])

    def forward(self, edge_feat, x_b_size):
        # e_f= []
        # for i in range(len(edge_feat)):
        #     import ipdb;ipdb.set_trace()
        #     # e_f.append(edge_feat[i].view([edge_feat[i].shape[0], edge_feat[i].shape[1], -1]))
        #     edge_feat[i] = self.module[i](edge_feat[i])
        #     # import ipdb;ipdb.set_trace()
        #     e_f.append(F.interpolate(edge_feat[i], size=(), mode='bilinear', align_corners=True))

        # for b in range(len(x_b_size)):
        #     for i in range(len(edge_feat)):
        #         edge_feat[0] = self.module[i](edge_feat[0])
        #         # import ipdb;
        #         # ipdb.set_trace()
        #         e_f.append(F.interpolate(edge_feat[0], size=(x_b_size[b][4-i]), mode='bilinear', align_corners=True))
        #         del edge_feat[0]
        #         gc.collect()
        #         torch.cuda.empty_cache()
        # del edge_feat
        # gc.collect()
        # torch.cuda.empty_cache()
        for i in range(len(x_b_size)):
            edge_feat[i] = self.module[i](edge_feat[i])
            edge_feat[i] = F.interpolate(edge_feat[i], size=x_b_size[4-i], mode='bilinear', align_corners=True)
        return edge_feat


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x, edge_f):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = edge_f.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(edge_f).view(batch_size, self.inter_channels, -1)

            del edge_f
            gc.collect()
            torch.cuda.empty_cache()

            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(edge_f).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))


        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z




# class Fusion(nn.Module):
#     def __init__(self,cfg):
#         super(Fusion,self).__init__()
#         self.in_channels=cfg.in_channels
#         self.out_channels=cfg.out_channels
#         self.conv_cfg=cfg.conv_cfg
#         self.norm_cfg=cfg.norm_cfg
#
#         self.convs=nn.ModuleList()
#         assert len(cfg.in_channels)==len(cfg.out_channels)
#
#         for i in range(len(cfg.in_channels)):
#             self.convs.append(ConvModule(
#                     self.in_channels[i],
#                     self.out_channels[i],
#                     1,
#                     padding=1,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=dict(type='GN', num_groups=self.norm_cfg[i], requires_grad=True),
#                     inplace=False
#                 ))
#
#     @auto_fp16()
#     def forward(self,feats, pre_masks):
#
#         assert feats[0].size(0)==len(pre_masks)
#         premasks=[]
#
#         for i in range(len(pre_masks)):
#             pre_masks[i]=pre_masks[i].permute(3,0,1,2)
#             # premasks.append(pre_masks[i])
#
#             h,w=pre_masks[i].shape[2:]
#             pre_masks[i]=torch.mean(pre_masks[i].reshape(1,-1,h,w),dim=1,keepdim=True)
#             # import ipdb;
#             # ipdb.set_trace()
#             # for conv in self.convs:
#             #     pre_masks[i]=conv(pre_masks[i])
#
#             premasks.append(pre_masks[i])
#
#         premasks = torch.cat(premasks, dim=0)
#         fusion_feat=[]
#
#         for i in range(len(feats)):
#             premasks=F.interpolate(premasks,feats[i].shape[2:],mode='bilinear', align_corners=True)
#             # import ipdb;
#             # ipdb.set_trace()
#             fusion_feat.append(premasks+feats[i])
#
#
#         return fusion_feat










class MBRM(nn.Module):
    """
    Mask based Boundary Refinement Module.
    """
    def __init__(self, cfg):
        super(MBRM, self).__init__()
        self.gamma = cfg.gamma
        self.kernel_size = cfg.kernel_size
        self.weight = nn.Parameter(torch.Tensor(1, 1, cfg.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.weight.data = torch.tensor([[[-0.2968, -0.2892, -0.2635, -3.2545, -0.1874, 2.5041, 0.0196, 0.0651, 0.0917]]])
        self.bias.data = torch.tensor([-1.9536])

    def forward(self, masks, boxes):
        """
        Refine boxes with masks.
        :param masks: Size: [num_dets, h, w]
        :param boxes: Size: [num_dets, 4], absolute coordinates
        :return:
        """
        num_dets, h, w = masks.size()
        if num_dets == 0:
            return None
        gamma = self.gamma

        horizon, _ = masks.max(dim=1)
        vertical, _ = masks.max(dim=2)

        gridx = torch.arange(w, device=masks.device, dtype=masks.dtype).view(1, -1)
        gridy = torch.arange(h, device=masks.device, dtype=masks.dtype).view(1, -1)

        sigma_h = ((boxes[:, 2] - boxes[:, 0]) * gamma).view(-1, 1)
        sigma_h.clamp_(min=1.0)
        sigma_v = ((boxes[:, 3] - boxes[:, 1]) * gamma).view(-1, 1)
        sigma_v.clamp_(min=1.0)

        sigma_h = 2 * sigma_h.pow(2)
        sigma_v = 2 * sigma_v.pow(2)

        p = int((self.kernel_size - 1) / 2)
        pl = F.conv1d(horizon.view(num_dets, 1, -1), self.weight, self.bias, padding=p).squeeze(1)
        pr = F.conv1d(horizon.view(num_dets, 1, -1), self.weight.flip(dims=(2, )), self.bias, padding=p).squeeze(1)
        pt = F.conv1d(vertical.view(num_dets, 1, -1), self.weight, self.bias, padding=p).squeeze(1)
        pb = F.conv1d(vertical.view(num_dets, 1, -1), self.weight.flip(dims=(2, )), self.bias, padding=p).squeeze(1)

        lweight = torch.exp(-(gridx - boxes[:, 0:1]).float().pow(2) / sigma_h)
        rweight = torch.exp(-(gridx - boxes[:, 2:3]).float().pow(2) / sigma_h)
        tweight = torch.exp(-(gridy - boxes[:, 1:2]).float().pow(2) / sigma_v)
        bweight = torch.exp(-(gridy - boxes[:, 3:4]).float().pow(2) / sigma_v)

        lweight = torch.where(lweight > 0.0044, lweight, lweight.new_zeros(1))
        rweight = torch.where(rweight > 0.0044, rweight, rweight.new_zeros(1))
        tweight = torch.where(tweight > 0.0044, tweight, tweight.new_zeros(1))
        bweight = torch.where(bweight > 0.0044, bweight, bweight.new_zeros(1))

        activate_func = torch.sigmoid

        pl = activate_func(pl) * lweight
        pr = activate_func(pr) * rweight
        pt = activate_func(pt) * tweight
        pb = activate_func(pb) * bweight

        dl = pl / pl.sum(dim=1, keepdim=True)
        dr = pr / pr.sum(dim=1, keepdim=True)
        dt = pt / pt.sum(dim=1, keepdim=True)
        db = pb / pb.sum(dim=1, keepdim=True)

        return [dl, dt, dr, db]

    def get_boxes(self, boundary_distribution, labels):
        if boundary_distribution is None:
            return labels.new_zeros(0, 5)
        (d_l, d_t, d_r, d_b) = boundary_distribution
        l = torch.argmax(d_l, dim=1).float()
        r = torch.argmax(d_r, dim=1).float()
        t = torch.argmax(d_t, dim=1).float()
        b = torch.argmax(d_b, dim=1).float()
        boxes = torch.stack([l, t, r, b], dim=1)

        return torch.cat([boxes, labels], -1)
