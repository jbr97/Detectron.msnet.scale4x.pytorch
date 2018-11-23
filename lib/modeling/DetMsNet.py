import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.resnet_weights_helper import convert_state_dict

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (DetMsNet75 ...
# ---------------------------------------------------------------------------- #

def DetMsNet75_stage4_body():
    return DetMsNet_stageX_body((3, 4, 6))


BN_MOMENTUM = 0.1

## temp
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.inplanes = inplanes
    self.planes = planes
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM) # change by jbr
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM) # change by jbr
    if self.inplanes != self.planes*self.expansion:
        self.downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes * self.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.planes * self.expansion,
                           momentum=BN_MOMENTUM),
        ) # change by jbr
    #self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    #if self.downsample is not None:
    # downsample condition change by jbr
    if self.inplanes != self.planes*self.expansion:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.inplanes = inplanes
    self.planes = planes
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM) # change by jbr
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM) # change by jbr
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM) # change by jbr
    self.relu = nn.ReLU(inplace=True)

    # downsample condition change by jbr
    if self.inplanes != self.planes*self.expansion:
        self.downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes * self.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.planes * self.expansion, momentum=BN_MOMENTUM),
        )
    #self.downsample = downsample
    #self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    #if self.downsample is not None:
    # downsample condition change by jbr
    if self.inplanes != self.planes*self.expansion:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ParamSample(nn.Module):
    """ Avg Pooling / Nearest Upsampling (2 times) followed by a point-wise convolution
        Args:
           op: str, pool or up
           inp_c: int, the number of input channels
           out_c: int, the number of output channels
    """

    def __init__(self, inp_c, out_c, factor, op='pool'):
        super(ParamSample, self).__init__()
        self.operator = op
        if op == 'pool':
            self.add_module('op', nn.AvgPool2d(
                kernel_size=factor, stride=factor, padding=0))
        elif op == 'up':
            self.add_module('op', nn.Upsample(
                scale_factor=factor, mode='nearest'))
        else:
            raise Exception('unknown op: %s' % op)
        self.add_module('conv', nn.Conv2d(
            inp_c, out_c,  kernel_size=1, padding=0, bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        if self.operator == 'pool':
            x = self.op(x)
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.op(x)
        return x


class XFusion(nn.Module):
    """ For each branch, aggregate information from all the branches.
        Args:
           inp_c: int, the number of input channels
    """

    def __init__(self, inp_c_list):
        super(XFusion, self).__init__()
        self.num_branch = len(inp_c_list)

        branches = []
        for i in range(self.num_branch):
            for j in range(self.num_branch):
                if i == j:
                    continue
                elif i < j:
                    branches.append(ParamSample(inp_c_list[i], inp_c_list[j],
                                                factor=2**(j-i), op='pool'))
                else:
                    branches.append(ParamSample(inp_c_list[i], inp_c_list[j],
                                                factor=2**(i-j), op='up'))
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        out = []
        for i in range(self.num_branch):
            out.append(x[i])
            for j in range(self.num_branch):
                if i == j:
                    continue
                elif i < j:
                    out[i] = out[i] + self.branches[j *
                                                    (self.num_branch-1)+i](x[j])
                else:
                    out[i] = out[i] + self.branches[j *
                                                    (self.num_branch-1)+i-1](x[j])
        return out



class MultiBranchNBlock(nn.Module):
    """ Multi branches and each branches consist of N residual blocks
        Args:
           n: the number of blocks for each branch
           inp_c_list: the list of input channels (the first 1x1 conv)
           inter_c_list: the list of intermediate channels (3x3 conv)
    """

    def __init__(self, n, block, inp_c_list, inter_c_list, expansion):
        super(MultiBranchNBlock, self).__init__()

        self.block = block
        self.num_branch = len(inp_c_list)
        self.expansion = expansion

        out_c_list = [inter_c_list[i] *
                      self.expansion for i in range(self.num_branch)]

        branches = []
        for i in range(self.num_branch):
            branches.append(nn.Sequential(OrderedDict(
                [('poolx%d_0' % (2**i), self.block(inp_c_list[i], inter_c_list[i]))])))
            for j in range(1, n[i]):
                branches[i].add_module('poolx%d_%d' % (2**i, j),
                                       self.block(out_c_list[i], inter_c_list[i]))
        self.branches = nn.ModuleList(branches)
        self.fuse = XFusion(inp_c_list=out_c_list)

    def forward(self, x):
        # x including the cooresponding multiple branches.
        # One or All, 0: full, 1: poolx2, 2: poolx4, 3: poolx8
        out = []
        for i in range(self.num_branch):
            out.append(self.branches[i](x[i]))
        out = self.fuse(out)
        return out


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

class DetMsNet_stageX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.channel_growth = cfg.MSNET.CHANNEL_GROWTH
        self.branch_depth = cfg.MSNET.BRANCH_DEPTH
        if cfg.MSNET.BLOCK_TYPE == 'BasicBlock':
            self.block = BasicBlock
            self.expansion = 1
        else:
            self.block = Bottleneck
            self.expansion = 4

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        '''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        '''

        # One-Branch Stage
        stage0_nblock = cfg.MSNET.NUM_BLOCKS[0]
        self.inplanes = cfg.MSNET.BASE_CHANNEL[0]
        self.stage0 = nn.Sequential(OrderedDict(
            [('stage0_0', Bottleneck(64, self.inplanes // 4))]))
        for i in range(1, stage0_nblock):
            self.stage0.add_module('stage0_%d' % i, Bottleneck(
                self.inplanes, self.inplanes // 4))
        '''
        layers = []
        layers.append(Bottleneck(64, self.inplanes // 4))
        for i in range(1, stage0_nblock):
            layers.append(Bottleneck(self.inplanes, self.inplanes // 4))
        self.layer1 = nn.Sequential(*layers)
        '''

        # Two-Branch Stage
        stage1_nblock = cfg.MSNET.NUM_BLOCKS[1]
        inp_c_list = [self.inplanes for i in range(2)]
        self.inplanes = cfg.MSNET.BASE_CHANNEL[1]
        inter_c_list = [np.int(self.inplanes//self.expansion *
                               self.channel_growth**i) for i in range(2)]
        self.stage1 = nn.Sequential(OrderedDict(
            [('stage1_0', MultiBranchNBlock(n=self.branch_depth[:2], block=self.block,
                                            inp_c_list=inp_c_list, inter_c_list=inter_c_list,
                                            expansion=self.expansion))]))
        inp_c_list = [np.int(self.inplanes*self.channel_growth**i)
                      for i in range(2)]
        for i in range(1, stage1_nblock):
            self.stage1.add_module('stage1_%d' % i, MultiBranchNBlock(
                n=self.branch_depth[:2], block=self.block, inp_c_list=inp_c_list,
                inter_c_list=inter_c_list, expansion=self.expansion))

        # Tri-Branch Stage
        stage2_nblock = cfg.MSNET.NUM_BLOCKS[2]
        inp_c_list = [np.sum(inp_c_list) for i in range(3)]
        self.inplanes = cfg.MSNET.BASE_CHANNEL[2]
        inter_c_list = [np.int(self.inplanes//self.expansion *
                               self.channel_growth**i) for i in range(3)]
        self.stage2 = nn.Sequential(OrderedDict([('stage2_0', MultiBranchNBlock(
            n=self.branch_depth[:3], block=self.block, inp_c_list=inp_c_list,
            inter_c_list=inter_c_list, expansion=self.expansion))]))
        inp_c_list = [np.int(self.inplanes*self.channel_growth**i)
                      for i in range(3)]
        for i in range(1, stage2_nblock):
            self.stage2.add_module('stage2_%d' % i, MultiBranchNBlock(
                n=self.branch_depth[:3], block=self.block, inp_c_list=inp_c_list,
                inter_c_list=inter_c_list, expansion=self.expansion))

        # Four-Branch Stage
        stage3_nblock = cfg.MSNET.NUM_BLOCKS[3]
        inp_c_list = [np.sum(inp_c_list) for i in range(4)]
        self.inplanes = cfg.MSNET.BASE_CHANNEL[3]
        inter_c_list = [np.int(self.inplanes//self.expansion*self.channel_growth**i)
                        for i in range(4)]
        self.stage3 = nn.Sequential(OrderedDict([('stage3_0', MultiBranchNBlock(
            n=self.branch_depth, block=self.block, inp_c_list=inp_c_list,
            inter_c_list=inter_c_list, expansion=self.expansion))]))
        inp_c_list = [np.int(self.inplanes*self.channel_growth**i)
                      for i in range(4)]
        for i in range(1, stage3_nblock):
            self.stage3.add_module('stage3_%d' % i, MultiBranchNBlock(
                n=self.branch_depth, block=self.block, inp_c_list=inp_c_list,
                inter_c_list=inter_c_list, expansion=self.expansion))
        # Upsampling
        self.upx2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upx4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upx8 = nn.Upsample(scale_factor=8, mode='nearest')

        # transition and align scale
        self.trans_4x = nn.Sequential(self.block(32, 64, stride=1), 
                                      self.block(64, 128, stride=1))
        self.trans_8x = nn.Sequential(self.block(64, 128, stride=1))
        self.trans_16x = nn.Sequential(self.block(128, 128, stride=1))
        self.trans_32x = nn.Sequential(self.block(256, 128, stride=1))

        
        self.dim_out = 512
        self.spatial_scale = 1. / 4
        self._init_modules()

        '''
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in
        '''

    def _init_modules(self):
        #assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        #assert cfg.RESNETS.FREEZE_AT <= self.convX
        #for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            #freeze_params(getattr(self, 'res%d' % i))
        freeze_params(self.stem)
        for i in range(0, 1):
            freeze_params(getattr(self, 'stage%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)
        self.apply(lambda m: freeze_params(m) if isinstance(m, nn.BatchNorm2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'res1.conv1.weight': 'conv1_w',
            'res1.bn1.weight': 'res_conv1_bn_s',
            'res1.bn1.bias': 'res_conv1_bn_b',
        }
        orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']
        '''

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)
        '''

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(1, 4):
            getattr(self, 'stage%d' % i).train(mode)
        self.apply(lambda m: freeze_params(m) if isinstance(m, nn.BatchNorm2d) else None)
    '''
    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x
    '''
    def forward(self, x):
        # stem
        x = self.stem(x)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        '''

        # One-Branch Stage
        #x = self.layer1(x)
        x = self.stage0(x)

        # Two-Branch Stage
        #x0 = x
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = [x, x1]
        x = self.stage1(x)

        # Tri-Branch Stage
        x1 = F.upsample(x[1], scale_factor=2, mode='nearest')
        x = torch.cat([x[0], x1], 1)
        x0 = x
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x2 = F.avg_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x = [x0, x1, x2]
        x = self.stage2(x)

        # Four-Branch Stage
        x1 = F.upsample(x[1], scale_factor=2, mode='nearest')
        x2 = F.upsample(x[2], scale_factor=4, mode='nearest')
        x = torch.cat([x[0], x1, x2], 1)
        x0 = x
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x2 = F.avg_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x3 = F.avg_pool2d(x2, kernel_size=2, stride=2, padding=0)
        x = [x0, x1, x2, x3]
        x = self.stage3(x)

        # Upsampling
        x0 = self.trans_4x(x[0])
        x1 = self.trans_8x(self.upx2(x[1]))
        x2 = self.trans_16x(self.upx4(x[2]))
        x3 = self.trans_32x(self.upx8(x[3]))

        y = torch.cat([x0, x1, x2, x3], 1)
        #y = self.transition(y)
        #y = self.final_layer(y)

        return y

    def init_weights(self, pretrained='',):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_dict, strict=False)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)



class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
          residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.RESNETS.NUM_GROUPS,
        downsample=downsample)

    return res_block


# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        mynn.AffineChannel2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', mynn.AffineChannel2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = mynn.AffineChannel2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
