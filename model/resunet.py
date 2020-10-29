# -*- coding: future_fstrings -*-
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm
import numpy as np
from model.residual_block import get_block
import torch.nn.functional as F
from torchviz import make_dot

class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  # ver2
  FC = None
  CH = None
  fcgf_extract = False
  max_pool = False
  sum_pool = False
  avg_pool = False
  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    # ver2
    FC = self.FC
    CH = self.CH


    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)
################################################################
    # ver2
    if FC :
      self.conv_1d = torch.nn.Conv1d(32, FC[0], 1)
      self.bn_1 = torch.nn.BatchNorm1d(FC[0])

    elif CH :
      self.conv_1d = torch.nn.Conv1d(32, CH[0]*2, 1)
      self.bn_1 = torch.nn.BatchNorm1d(CH[0]*2)
      self.fc1 = torch.nn.Linear(CH[0]*2, CH[0])
      self.bn_2 = torch.nn.BatchNorm1d(CH[0])

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)
    m0=out.F
    l0 = out.decomposed_features[0].shape[0]

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)
    m1=out_s4_tr.F
    l1 = out_s4_tr.decomposed_features[0].shape[0]

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)
    m2=out_s2_tr.F
    l2 = out_s2_tr.decomposed_features[0].shape[0]

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)

    out = self.final(out)

    #print(out.F.shape)
    if self.normalize_feature:
      out = ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    # print(len(out.decomposed_features))
    #
    # print(out.coords_key)
    # print(out.coords_man)
    m3=out.F
    l3 = out.decomposed_features[0].shape[0]

    ls = [l0, l1, l2, l3]
    if self.fcgf_extract :
      return m0, m1, m2, m3, ls

    # if self.fcgf_extract :
    #   return out.F, out.decomposed_features[0].shape[0]

    if self.max_pool :
      out = torch.max(out.F, 0)[0]
      print('resunet - max_pool shape : ', out.shape)
      return out
    if self.sum_pool :
      out = torch.sum(out.F, 0)
      print('resunet - max_pool shape : ', out.shape)
      return out
    if self.avg_pool :
      out = torch.sum(out.F, 0) / out.decomposed_features[0].shape[0]
      print('resunet - max_pool shape : ', out.shape)
      return out
################################################################
    # ver2
    if self.FC :
      # save index points per batch's start point
      n_per_batch = []
      for i in range(len(out.decomposed_features)):
        n_per_batch.append(out.decomposed_features[i].shape[0])
      # shape : [batch*N, 32]
      out = out.F.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out = out.view(1, out.shape[-1], -1)
      # shape : [1, 32, batch*N]
      out = F.relu(self.bn_1(self.conv_1d(out)))
      out = out.squeeze(0).view(out.shape[-1], -1)
      # shape : [batch*N, 128]
      temp_out=[]
      ind_point = 0
      for i in range(len(n_per_batch)) :
        if i==0 :
          ind_start = 0
          ind_end = n_per_batch[i]
        else :
          ind_start = ind_point
          ind_end = ind_point + n_per_batch[i]

        temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
        ind_point = ind_end
      # shape : [128] * batch
      result = torch.cat(temp_out, 0)
      '''
      마지막에 normalize를 하지 않는다면?
      '''
      result = F.normalize(result, p=2)

      # print('resunet.py - output shape : ', out.shape)
      print(result)
      print(result.shape)
      return result

    #
    # if self.FC :
    #   out = out.F.unsqueeze(0)
    #   out = out.view(1, out.shape[-1], -1)
    #   out = F.relu(self.bn_1(self.conv_1d(out)))
    #   out = out.view(1, out.shape[-1], -1)
    #   out = torch.max(out, 1)[0]
    #   out = F.normalize(out, p=2)
    #
    #   return out
    elif self.CH :
      # [batch*N, 32]
      out.decomposed_features
      out = out.F.unsqueeze(0)
      # [1, batch*N, 32]
      out = out.view(1, out.shape[-1], -1)
      # [1, 32, batch*N]
      print('test'*20)
      out = self.conv_1d(out)
      print(out)
      out = self.bn_1(out)
      print(out)
      out = F.relu(out)
      print(out)
      print('test' * 20)
      #out = F.relu(self.bn_1(out))
      #out = F.relu(self.bn_1(self.conv_1d(out)))
      print(out.shape)
      # out = F.relu(self.bn_1(self.conv_1d(out)))
      assert 0

      out = self.linear(input.F)
      return SparseTensor(
        output,
        coords_key=input.coords_key,
        coords_manager=input.coords_man)

      print('0'*20)
      out = out.F.unsqueeze(0)
      print(out.shape)
      out = out.view(1, out.shape[-1], -1)
      print(out.shape)
      out = F.relu(self.conv_1d(out))
      print(out.shape)
      out = out.view(1, out.shape[-1], -1)
      print(out.shape)
      out = torch.max(out, 1)[0]
      print(out.shape)
      out = F.relu(self.fc1(out))
      print(out.shape)
      print('0' * 20)
      out = F.normalize(out, p=2)
      assert 0

    # elif self.CH :
    #   print('0'*20)
    #   out = out.F.unsqueeze(0)
    #   print(out.shape)
    #   out = out.view(1, out.shape[-1], -1)
    #   print(out.shape)
    #   out = F.relu(self.conv_1d(out))
    #   print(out.shape)
    #   out = out.view(1, out.shape[-1], -1)
    #   print(out.shape)
    #   out = torch.max(out, 1)[0]
    #   print(out.shape)
    #   out = F.relu(self.fc1(out))
    #   print(out.shape)
    #   print('0' * 20)
    #   out = F.normalize(out, p=2)
    #   assert 0

      return out
    ################################################# 일단 batchnorm 필요없을듯

    else :
      return out.F

    # if self.FC :
    #   out = out.F.unsqueeze(0)
    #   out = out.view(1, out.shape[-1], -1)
    #   out = F.relu(self.bn_1(self.conv_1d(out)))
    #   # out = F.relu(self.conv_1d(out))
    #   out = out.view(1, out.shape[-1], -1)
    #   return out
    # else :
    #   return out.F

class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'

############################################################
# ver2
class ResUNetMLP2(ResUNetBN2C):
  FC = [128]

class ResUNetMLP2B(ResUNetBN2C):
  FC = [256]

class ResUNetMLP2C(ResUNetBN2C):
  FC = [512]

############################################################
# ver3
class ResUNetCH2(ResUNetBN2C) :
  CH = [128]

class ResUNetBN2C_max(ResUNetBN2C) :
  max_pool = True

class ResUNetBN2C_sum(ResUNetBN2C) :
  sum_pool = True

class ResUNetBN2C_avg(ResUNetBN2C) :
  avg_pool = True
############################################################
# ver5 - for save fcgf & length

class ResUNetBN2C_extract(ResUNetBN2C) :
  fcgf_extract=True

############################################################
# ver5 - load fcgf

class FCGF_MLP3(torch.nn.Module) :
  FCGF = [64, 256]
  FC = []
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_MLP3, self).__init__()
    FCGF = self.FCGF
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, FCGF[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FCGF[0])
    self.conv1d_2 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(FCGF[1])
    if FC :
      self.fc_1 = torch.nn.Linear(FCGF[-1], FC[0], 1)
      self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])
      self.fc_2 = torch.nn.Linear(FC[0], FC[1], 1)
      self.fc_bn_2 = torch.nn.BatchNorm1d(FC[1])


  def forward(self, x, length):
    # out_s1 = self.conv1(x)
    # out_s1 = self.norm1(out_s1)
    # out_s1 = self.block1(out_s1)
    # out = MEF.relu(out_s1)
    # save index points per batch's start point

    # save index points per batch's start point
    n_per_batch = length

    # x shape : [batch*N, 32]
    out = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out = out.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out = F.relu(self.bn_1(self.conv1d_1(out)))
    out = F.relu(self.bn_2(self.conv1d_2(out)))
    out = out.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 256]
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      temp_out.append(torch.mean(out[ind_start : ind_end], 0).unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]

    result = torch.cat(temp_out, 0)
    if self.FC :
      result = F.relu(self.fc_bn_1(self.fc_1(result)))
      result = self.fc_bn_2(self.fc_2(result))

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result

############################################################
class FCGF_MLP3B(FCGF_MLP3) :
  FC = [64, 32]
############################################################

class FCGF_point_att_demo(torch.nn.Module) :
  FC = [1]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_demo, self).__init__()
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    if len(FC) > 1 :
      self.conv1d_2 = torch.nn.Conv1d(32, FC[1], 1)
      self.bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = (self.bn_1(self.conv1d_1(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]

    if len(self.FC) > 1:
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = (self.bn_2(self.conv1d_2(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * out1[ind_start : ind_end]
      # temp = torch.sum(temp, 0).unsqueeze(0)
      temp_out.append(out1[ind_start : ind_end])
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end
    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    # result = F.normalize(result, p=2)

    return result

############################################################
class FCGF_point_att3(torch.nn.Module) :
  ATT = [16, 1]
  FCGF = None

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att3, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF

    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    if FCGF :
      self.conv1d_3 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_3 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_4 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = self.bn_2(self.conv1d_2(out1))
    out1 = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]
    if self.FCGF :
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_3(self.conv1d_3(out2)))
      out2 = (self.bn_4(self.conv1d_4(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * out1[ind_start : ind_end]
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result
############################################################
class FCGF_point_att4_sft(torch.nn.Module) :
  ATT = [16, 8, 1]
  FCGF = [64, 128]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att4_sft, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF



    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    self.conv1d_3 = torch.nn.Conv1d(ATT[1], ATT[2], 1)
    self.bn_3 = torch.nn.BatchNorm1d(ATT[2])
    if FCGF :
      self.conv1d_4 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_5 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_5 = torch.nn.BatchNorm1d(FCGF[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = F.relu(self.bn_2(self.conv1d_2(out1)))
    out1 = F.relu(self.bn_3(self.conv1d_3(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)


    # shape : [batch*N, 1]
    if self.FCGF :
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_4(self.conv1d_4(out2)))
      out2 = (self.bn_5(self.conv1d_5(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * F.softmax(out1[ind_start : ind_end], 0)
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result
############################################################
class FCGF_point_att4_fc(torch.nn.Module) :
  ATT = [16, 8, 1]
  FCGF = [64, 128]
  FC = [64, 256]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att4_fc, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF
    FC = self.FC


    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    self.conv1d_3 = torch.nn.Conv1d(ATT[1], ATT[2], 1)
    self.bn_3 = torch.nn.BatchNorm1d(ATT[2])
    if FCGF :
      self.conv1d_4 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_5 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_5 = torch.nn.BatchNorm1d(FCGF[1])
    if FC :
      self.fc_1 = torch.nn.Linear(128, FC[0], 1)
      self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])
      self.fc_2 = torch.nn.Linear(FC[0], FC[1], 1)
      self.fc_bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = F.relu(self.bn_2(self.conv1d_2(out1)))
    out1 = F.relu(self.bn_3(self.conv1d_3(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)

    # shape : [batch*N, 1]
    if self.FCGF :
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_4(self.conv1d_4(out2)))
      out2 = F.relu(self.bn_5(self.conv1d_5(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * F.softmax(out1[ind_start : ind_end], 0)
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)
    result = F.relu((self.fc_bn_1(self.fc_1(result))))
    result = ((self.fc_bn_2(self.fc_2(result))))
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result
############################################################
class FCGF_point_att3_sft(torch.nn.Module) :
  ATT = [16, 1]
  FCGF = None

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att3_sft, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF

    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    if FCGF :
      self.conv1d_3 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_3 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_4 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = F.relu(self.bn_2(self.conv1d_2(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)

    # shape : [batch*N, 1]
    if self.FCGF :
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_3(self.conv1d_3(out2)))
      out2 = (self.bn_4(self.conv1d_4(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * F.softmax(out1[ind_start : ind_end], 0)
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result
############################################################
class FCGF_point_att3_sft_7000(torch.nn.Module) :
  ATT = [16, 8, 1]
  FC = [1024, 256]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att3_sft_7000, self).__init__()
    ATT = self.ATT
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    self.conv1d_3 = torch.nn.Conv1d(ATT[1], ATT[2], 1)
    self.bn_3 = torch.nn.BatchNorm1d(ATT[2])
    self.fc_1 = torch.nn.Linear(2000*32, FC[0])
    self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.fc_2 = torch.nn.Linear(FC[0], FC[1])
    self.fc_bn_2 = torch.nn.BatchNorm1d(FC[1])


  def forward(self, x, length):

    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = F.relu(self.bn_2(self.conv1d_2(out1)))
    out1 = F.relu(self.bn_3(self.conv1d_3(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)

    # shape : [batch*N, 1]
    out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      katt = out1[ind_start : ind_end][:2000]


      temp = out2[ind_start : ind_end][:2000] * F.softmax(katt, 0)
      temp_out.append(torch.flatten(temp).unsqueeze(0))

      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)
    result = F.relu(self.fc_bn_1(self.fc_1(result)))
    result = F.relu(self.fc_bn_2(self.fc_2(result)))
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result

############################################################
class FCGF_point_att3_sft_demo(torch.nn.Module) :
  ATT = [16, 1]
  FCGF = None

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att3_sft_demo, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF

    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    if FCGF :
      self.conv1d_3 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_3 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_4 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = F.relu(self.bn_2(self.conv1d_2(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)

    # shape : [batch*N, 1]
    if self.FCGF :
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_3(self.conv1d_3(out2)))
      out2 = (self.bn_4(self.conv1d_4(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * F.softmax(out1[ind_start : ind_end], 0)
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(out1[ind_start : ind_end])
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    # result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result
############################################################
class FCGF_point_att3_fc(torch.nn.Module):
  ATT = [16, 1]
  FCGF = None
  FC = [64, 256]
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att3_fc, self).__init__()
    ATT = self.ATT
    FCGF = self.FCGF
    FC = self.FC
    self.conv1d_1 = torch.nn.Conv1d(32, ATT[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(ATT[0])
    self.conv1d_2 = torch.nn.Conv1d(ATT[0], ATT[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(ATT[1])
    self.fc_1 = torch.nn.Linear(32, FC[0])
    self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.fc_2 = torch.nn.Linear(FC[0], FC[1])
    self.fc_bn_2 = torch.nn.BatchNorm1d(FC[1])
    if FCGF:
      self.conv1d_3 = torch.nn.Conv1d(32, FCGF[0], 1)
      self.bn_3 = torch.nn.BatchNorm1d(FCGF[0])
      self.conv1d_4 = torch.nn.Conv1d(FCGF[0], FCGF[1], 1)
      self.bn_4 = torch.nn.BatchNorm1d(FCGF[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = F.relu(self.bn_1(self.conv1d_1(out1)))
    out1 = self.bn_2(self.conv1d_2(out1))
    out1 = out1.squeeze(0).transpose(1, 0)

    # shape : [batch*N, 1]
    if self.FCGF:
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = F.relu(self.bn_3(self.conv1d_3(out2)))
      out2 = F.relu(self.bn_4(self.conv1d_4(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else:
      out2 = x
    temp_out = []
    ind_point = 0
    for i in range(len(n_per_batch)):
      if i == 0:
        ind_start = 0
        ind_end = n_per_batch[i]
      else:
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * F.softmax(out1[ind_start : ind_end], 0)
      temp = torch.mean(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [128]
    result = torch.cat(temp_out, 0)
    result = F.relu((self.fc_bn_1(self.fc_1(result))))
    result = ((self.fc_bn_2(self.fc_2(result))))
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result


############################################################
class FCGF_point_att2(torch.nn.Module) :
  FC = [1]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att2, self).__init__()
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    if len(FC) > 1:
      self.conv1d_2 = torch.nn.Conv1d(32, FC[1], 1)
      self.bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = (self.bn_1(self.conv1d_1(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]
    if len(self.FC) > 1:
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = (self.bn_2(self.conv1d_2(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]
    else :
      out2 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      temp = out2[ind_start : ind_end] * out1[ind_start : ind_end]
      temp = torch.sum(temp, 0).unsqueeze(0)
      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end
    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result

############################################################

class FCGF_point_att2B(FCGF_point_att2) :
  FC = [1, 128]

class FCGF_point_att2C(FCGF_point_att2) :
  FC = [1, 256]

class FCGF_point_att2D(FCGF_point_att2) :
  FC = [1, 512]

############################################################


class FCGF_point_att2_ican(torch.nn.Module) :
  FC = [1]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att2_ican, self).__init__()
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])

    if len(FC) > 1:
      self.conv1d_2 = torch.nn.Conv1d(32, FC[1], 1)
      self.bn_2 = torch.nn.BatchNorm1d(FC[1])

      self.conv1d_3 = torch.nn.Conv1d(32, FC[1], 1)
      self.bn_3 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = (self.bn_1(self.conv1d_1(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]
    if len(self.FC) > 1:
      out2 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out2 = out2.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out2 = (self.bn_2(self.conv1d_2(out2)))
      out2 = out2.squeeze(0).transpose(1, 0)
      # shape : [batch*N, 128]

      out3 = x.unsqueeze(0)
      # shape : [1, batch*N, 32]
      out3 = out3.transpose(2, 1)
      # shape : [1, 32, batch*N]
      out3 = (self.bn_3(self.conv1d_3(out3)))
      out3 = out3.squeeze(0).transpose(1, 0)
    else :
      out2 = x
      out3 = x
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      sft = out2[ind_start : ind_end] * out1[ind_start : ind_end]
      sft = torch.mean(sft, 1).unsqueeze(1)
      sft = F.softmax(sft, 0)

      temp = out3[ind_start : ind_end] * sft
      temp = torch.mean(temp, 0).unsqueeze(0)

      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end
    # shape : batch * [128]
    result = torch.cat(temp_out, 0)

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result




############################################################

class FCGF_point_att2_icanB(FCGF_point_att2_ican) :
  FC = [1, 128]

class FCGF_point_att2_icanC(FCGF_point_att2_ican) :
  FC = [1, 256]

class FCGF_point_att2_icanD(FCGF_point_att2_ican) :
  FC = [1, 512]

############################################################
class FCGF_point_att2_ican_fc(torch.nn.Module) :
  FC = [1, 64]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att2_ican_fc, self).__init__()
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])

    if len(FC) > 1:
      self.fc_1 = torch.nn.Linear(32, FC[1], 1)
      self.bn_2 = torch.nn.BatchNorm1d(FC[1])


  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = (self.bn_1(self.conv1d_1(out1)))
    out1 = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]

    out2 = x
    out3 = x

    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]
      sft = out2[ind_start : ind_end] * out1[ind_start : ind_end]
      sft = torch.mean(sft, 1).unsqueeze(1)
      sft = F.softmax(sft, 0)

      temp = out3[ind_start : ind_end] * sft
      temp = torch.mean(temp, 0).unsqueeze(0)


      temp_out.append(temp)
      # temp_out.append(torch.max(out[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end
    # shape : batch * [32]
    result = torch.cat(temp_out, 0)
    result = (self.bn_2(self.fc_1(result)))

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    # result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result




############################################################


class FCGF_RP_AVG(torch.nn.Module) :
  FC = [256]
  topk = 1024
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_RP_AVG, self).__init__()
    FC = self.FC
    topk = self.topk
    self.conv1d_1 = torch.nn.Conv1d(32, 1, 1)


  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = self.conv1d_1(out1)
    att = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]

    # att = F.softmax(att, 1)

    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start : ind_end]
      # temp_att shape : [N, 1]
      temp_att = att[ind_start : ind_end]
      _, s_idx = torch.sort(temp_att, dim=0, descending=True)
      zero_idx = s_idx[self.topk:]
      one_idx = s_idx[:self.topk]
      temp_att[zero_idx]=0
      temp_att[one_idx]=1
      ranked_point = temp_input*temp_att
      # s_idx = s_idx.squeeze(1)
      # ranked_point = temp_input[s_idx[:self.topk]]
      # ranked_point shape : [K=1024, 32]

      temp_out.append(torch.mean(ranked_point, 0).unsqueeze(0))

      ind_point = ind_end

    result = torch.cat(temp_out, 0)

    # result shape : [batch, 32]
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)
    print(self.conv1d_1.weight)
    # print('resunet.py - output shape : ', out.shape)

    return result




############################################################

############################################################


class FCGF_RP_FC(torch.nn.Module):
  FC = [256]
  topk = 1024

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_RP_FC, self).__init__()
    FC = self.FC
    topk = self.topk
    self.conv1d_1 = torch.nn.Conv1d(32, 1, 1)
    self.bn_1 = torch.nn.BatchNorm1d(1)
    self.fc_1 = torch.nn.Linear(topk * 32, FC[0])
    self.bn_2 = torch.nn.BatchNorm1d(FC[0])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    out1 = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    out1 = out1.transpose(2, 1)
    # shape : [1, 32, batch*N]
    out1 = (self.bn_1(self.conv1d_1(out1)))
    att = out1.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1]

    # att = F.softmax(att, 1)

    temp_out = []
    ind_point = 0
    for i in range(len(n_per_batch)):
      if i == 0:
        ind_start = 0
        ind_end = n_per_batch[i]
      else:
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start: ind_end]
      # temp_att shape : [N, 1]
      temp_att = att[ind_start: ind_end]
      _, s_idx = torch.sort(temp_att, dim=0, descending=True)

      s_idx = s_idx.squeeze(1)
      ranked_point = temp_input[s_idx[:self.topk]]
      # ranked_point shape : [K=1024, 32]
      temp_out.append(torch.flatten(ranked_point).unsqueeze(0))
      ind_point = ind_end

    flat = torch.cat(temp_out, 0)
    # result shape : [batch, K*32]
    result = self.bn_2(self.fc_1(flat))
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)

    return result


############################################################
############################################################


class FCGF_point_att_k(torch.nn.Module) :
  FC = [256, 1024]
  topk = 1024
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_k, self).__init__()
    FC = self.FC
    topk = self.topk
    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.conv1d_2 = torch.nn.Conv1d(FC[0], FC[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(FC[1])


  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    att_k = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    att_k = att_k.transpose(2, 1)
    # shape : [1, 32, batch*N]
    att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
    att_k = F.relu(self.bn_2(self.conv1d_2(att_k)))
    att_k = att_k.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 1024]

    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start : ind_end]
      # temp_att shape : [N, 1024]
      temp_att = F.softmax(att_k[ind_start : ind_end], 0)

      out = (torch.matmul(temp_att.T, temp_input)) / n_per_batch[i]
      # out shape : [1024, 32]
      temp_out.append(torch.mean(out, 0).unsqueeze(0))

      ind_point = ind_end

    result = torch.cat(temp_out, 0)

    # result shape : [batch, 32]
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)
    # print('resunet.py - output shape : ', out.shape)
    return result

############################################################

############################################################


class FCGF_point_att_k_new(torch.nn.Module) :
  K_MLP = [128, 512]
  FC = [256]
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_k_new, self).__init__()
    K_MLP = self.K_MLP
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, K_MLP[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(K_MLP[0])
    self.conv1d_2 = torch.nn.Conv1d(K_MLP[0], K_MLP[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(K_MLP[1])
    self.fc_1 = torch.nn.Linear(32 * K_MLP[-1], FC[0])
    self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])


  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    att_k = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    att_k = att_k.transpose(2, 1)
    # shape : [1, 32, batch*N]
    if len(n_per_batch) == 2 :
      att_k = F.relu(self.conv1d_1(att_k))
      att_k = F.relu(self.conv1d_2(att_k))
    else :
      att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
      att_k = F.relu(self.bn_2(self.conv1d_2(att_k)))
    att_k = att_k.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 512]

    temp_out=[]
    gp_num=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start : ind_end]
      # temp_att shape : [N, 512]
      temp_att = att_k[ind_start : ind_end]

      # print((temp_att[:,0]!=0).sum())

      out = (torch.matmul(temp_att.T, temp_input)) / n_per_batch[i]
      # out shape : [512, 32]
      gp_num.append(torch.sum(temp_att, 0).unsqueeze(0))
      # gp_num[i] shape : [512]
      temp_out.append(torch.flatten(out).unsqueeze(0))
      # out = torch.flatten(out).unsqueeze(0)
      # out = self.fc_1 (out)
      # temp_out.append(out)
      ind_point = ind_end
    gp = torch.cat(gp_num, 0)

    result = torch.cat(temp_out, 0)
    if len(n_per_batch) == 2 :
      result = self.fc_1(result)
    else :
      result = self.fc_bn_1(self.fc_1(result))


    # result shape : [batch, 32]
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)
    # print('resunet.py - output shape : ', out.shape)
    ## regularization ##
    return result, gp, np.array(n_per_batch).reshape(-1, 1)
    # return result

############################################################

############################################################


class FCGF_AVG_multi(torch.nn.Module) :
  K_MLP = [128, 512]
  FC = [256]
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_AVG_multi, self).__init__()
    K_MLP = self.K_MLP
    FC = self.FC

    self.fc_1 = torch.nn.Linear(32 * K_MLP[-1], FC[0])
    self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])


  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    att_k = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    att_k = att_k.transpose(2, 1)
    # shape : [1, 32, batch*N]
    if len(n_per_batch) == 2 :
      att_k = F.relu(self.conv1d_1(att_k))
      att_k = F.relu(self.conv1d_2(att_k))
    else :
      att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
      att_k = F.relu(self.bn_2(self.conv1d_2(att_k)))
    att_k = att_k.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 512]

    temp_out=[]
    gp_num=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start : ind_end]
      # temp_att shape : [N, 512]
      temp_att = att_k[ind_start : ind_end]

      # print((temp_att[:,0]!=0).sum())

      out = (torch.matmul(temp_att.T, temp_input)) / n_per_batch[i]
      # out shape : [512, 32]
      gp_num.append(torch.sum(temp_att, 0).unsqueeze(0))
      # gp_num[i] shape : [512]
      temp_out.append(torch.flatten(out).unsqueeze(0))
      # out = torch.flatten(out).unsqueeze(0)
      # out = self.fc_1 (out)
      # temp_out.append(out)
      ind_point = ind_end
    gp = torch.cat(gp_num, 0)

    result = torch.cat(temp_out, 0)
    if len(n_per_batch) == 2 :
      result = self.fc_1(result)
    else :
      result = self.fc_bn_1(self.fc_1(result))


    # result shape : [batch, 32]
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)
    # print('resunet.py - output shape : ', out.shape)
    ## regularization ##
    return result, gp, np.array(n_per_batch).reshape(-1, 1)
    # return result

############################################################



class FCGF_point_att_k_new2(torch.nn.Module) :
  K_MLP = [128, 512]
  FC = []
  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_k_new2, self).__init__()
    K_MLP = self.K_MLP
    FC = self.FC

    self.conv1d_1 = torch.nn.Conv1d(32, K_MLP[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(K_MLP[0])
    self.conv1d_2 = torch.nn.Conv1d(K_MLP[0], K_MLP[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(K_MLP[1])



  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    att_k = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    att_k = att_k.transpose(2, 1)
    # shape : [1, 32, batch*N]
    att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
    att_k = F.relu(self.bn_2(self.conv1d_2(att_k)))
    att_k = att_k.squeeze(0).transpose(1, 0)
    # shape : [batch*N, 512]

    temp_out=[]
    gp_num=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start : ind_end]
      # temp_att shape : [N, 512]
      temp_att = att_k[ind_start : ind_end]

      # print((temp_att[:,0]!=0).sum())

      out = (torch.matmul(temp_att.T, temp_input)) / n_per_batch[i]
      # out shape : [512, 32]
      gp_num.append(torch.sum(temp_att, 0).unsqueeze(0))
      # gp_num[i] shape : [512]
      temp_out.append(torch.mean(out, 0).unsqueeze(0))

      ind_point = ind_end
    gp = torch.cat(gp_num, 0)

    result = torch.cat(temp_out, 0)

    # result shape : [batch, 32]
    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)
    # print('resunet.py - output shape : ', out.shape)
    return result, gp, np.array(n_per_batch).reshape(-1, 1)

############################################################
############################################################


class FCGF_point_att_k_fc(torch.nn.Module):
  FC = [256, 1024]
  topk = 1024

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_k_fc, self).__init__()
    FC = self.FC
    topk = self.topk
    self.conv1d_1 = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.conv1d_2 = torch.nn.Conv1d(FC[0], FC[1], 1)
    self.bn_2 = torch.nn.BatchNorm1d(FC[1])
    self.fc_1 = torch.nn.Linear(32*FC[1], FC[0])
    self.bn_3 = torch.nn.BatchNorm1d(FC[0])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # input1 shape : [batch*N, 32]
    input1 = x
    # x shape : [batch*N, 32]
    att_k = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    att_k = att_k.transpose(2, 1)
    # shape : [1, 32, batch*N]
    att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
    att_k = F.relu(self.bn_2(self.conv1d_2(att_k)))
    att_k = att_k.squeeze(0).transpose(1, 0)
    '''
    relu를 쓰는게 좋을까?
    '''
    # att_k = F.relu(self.bn_1(self.conv1d_1(att_k)))
    # shape : [batch*N, 1024]

    temp_out = []
    ind_point = 0

    for i in range(len(n_per_batch)):
      if i == 0:
        ind_start = 0
        ind_end = n_per_batch[i]
      else:
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      temp_input = input1[ind_start: ind_end]
      # temp_att shape : [N, 256]
      temp_att = F.softmax(att_k[ind_start : ind_end], 0)

      out = (torch.matmul(temp_att.T, temp_input)) / n_per_batch[i]
      # out shape : [1024, 32]
      temp_out.append(torch.flatten(out).unsqueeze(0))

      ind_point = ind_end

    # result shape : [batch, 256*32]
    out2 = torch.cat(temp_out, 0)

    result = self.bn_3(self.fc_1(out2))
    # result shape : [batch, 256]

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    # print('resunet.py - output shape : ', out.shape)
    return result
############################################################
############################################################


class FCGF_point_att_k_final(torch.nn.Module):
  POINT_ATT = [16, 1]
  CHANNEL_ATT = [16, 32]
  K_MLP = [128, 512]
  FC = [512, 256]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_point_att_k_final, self).__init__()
    POINT_ATT = self.POINT_ATT
    CHANNEL_ATT = self.CHANNEL_ATT
    K_MLP = self.K_MLP
    FC = self.FC


    # Point attention
    self.po_conv1d_1 = torch.nn.Conv1d(32, POINT_ATT[0], 1)
    self.po_bn_1 = torch.nn.BatchNorm1d(POINT_ATT[0])
    self.po_conv1d_2 = torch.nn.Conv1d(POINT_ATT[0], POINT_ATT[1], 1)
    self.po_bn_2 = torch.nn.BatchNorm1d(POINT_ATT[1])

    # Channel attention
    self.ch_fc_1 = torch.nn.Linear(32, CHANNEL_ATT[0])
    # self.ch_bn_1 = torch.nn.BatchNorm1d(CHANNEL_ATT[0])
    self.ch_fc_2 = torch.nn.Linear(CHANNEL_ATT[0], CHANNEL_ATT[1])
    # self.ch_bn_2 = torch.nn.BatchNorm1d(CHANNEL_ATT[1])

    # K_MLP
    self.k_conv1d_1 = torch.nn.Conv1d(32, K_MLP[0], 1)
    self.k_bn_1 = torch.nn.BatchNorm1d(K_MLP[0])
    self.k_conv1d_2 = torch.nn.Conv1d(K_MLP[0], K_MLP[1], 1)
    self.k_bn_2 = torch.nn.BatchNorm1d(K_MLP[1])

    # FC
    self.fc_1 = torch.nn.Linear(32 * K_MLP[1], FC[0])
    self.fc_bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.fc_2 = torch.nn.Linear(FC[0], FC[1])
    self.fc_bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # print(x[:5])
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]
    p_att = x.unsqueeze(0)
    # shape : [1, batch*N, 32]
    p_att = p_att.transpose(2, 1)
    # shape : [1, 32, batch*N]
    p_att = F.relu(self.po_bn_1(self.po_conv1d_1(p_att)))
    p_att = F.relu(self.po_bn_2(self.po_conv1d_2(p_att)))
    p_att = p_att.squeeze(0).transpose(1, 0)

    # p_att shape : [batch*N, 1]


    ind_point = 0
    flt_fcgf=[]
    for i in range(len(n_per_batch)):
      if i == 0:
        ind_start = 0
        ind_end = n_per_batch[i]
      else:
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      # temp_input shape : [N, 32]
      input = x[ind_start: ind_end]
      # temp_att shape : [N, 256]
      _p_att = F.softmax(p_att[ind_start : ind_end], 0)

      _c_att = torch.mean(input, 0).unsqueeze(0)
      _c_att = F.relu(self.ch_fc_1(_c_att))
      _c_att = F.relu(self.ch_fc_2(_c_att))
      _c_att = F.softmax(_c_att, 1)

      flt_input = input * _p_att * _c_att
      flt_fcgf.append(flt_input)
      ind_point = ind_end

    flt_fcgf = torch.cat(flt_fcgf, 0)
    flt_fcgf=flt_fcgf*1e+7
    # flt_fcgf shape : [N*batch, 32]
    # print(flt_fcgf[:5])
    k_gd = flt_fcgf.unsqueeze(0)
    k_gd = k_gd.transpose(2, 1)
    k_gd = F.relu(self.k_bn_1(self.k_conv1d_1(k_gd)))
    # print(k_gd[0, 0])
    k_gd = F.relu(self.k_bn_2(self.k_conv1d_2(k_gd)))
    # print(k_gd[0, 0])
    k_gd = k_gd.squeeze(0).transpose(1, 0)
    # k_gd shape : [N*batch, 512]
    # print(k_gd.T[0, :1000])


    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)):
      if i == 0:
        ind_start = 0
        ind_end = n_per_batch[i]
      else:
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      temp_input = flt_fcgf[ind_start: ind_end]
      temp_gd = F.softmax(k_gd[ind_start: ind_end], 0)
      out = (torch.matmul(temp_gd.T, temp_input)) / n_per_batch[i]

      temp_out.append(torch.flatten(out).unsqueeze(0))
      ind_point = ind_end


    out2 = torch.cat(temp_out, 0)

    # print(out2[0])
    result = F.relu(self.fc_bn_1(self.fc_1(out2)))
    # print(result[0])
    result = self.fc_bn_2(self.fc_2(result))
    # print(result[0])
    # result shape : [batch, 256]

    result = F.normalize(result, p=2)

    return result
############################################################


class FCGF_MLP2(torch.nn.Module) :
  FC = [128]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_MLP2, self).__init__()
    FC = self.FC

    self.conv_1d = torch.nn.Conv1d(32, FC[0], 1)
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]

    # out = x.unsqueeze(0)
    # # shape : [1, batch*N, 32]
    # out = out.view(1, out.shape[-1], -1)
    # # shape : [1, 32, batch*N]
    # out = F.relu(self.bn_1(self.conv_1d(out)))
    # out = out.squeeze(0).view(out.shape[-1], -1)
    # # shape : [batch*N, 128]
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      temp_out.append(torch.max(x[ind_start : ind_end], 0)[0].unsqueeze(0))
      ind_point = ind_end

    # shape : batch * [32]
    out = torch.cat(temp_out, 0)
    out = out.unsqueeze(0)
    # shape : [1, batch, 32]
    out = out.transpose(2, 1)
    # shape : [1, 32, batch]
    out = F.relu(self.bn_1(self.conv_1d(out)))
    # shape : [1, 128, batch]
    result = out.squeeze(0).transpose(1, 0)
    # shape : [batch, 128]

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    # out = F.normalize(out, p=2)

    return result

############################################################

class FCGF_AVG2(torch.nn.Module) :
  FC = [64, 128]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_AVG2, self).__init__()
    FC = self.FC

    self.fc_1 = torch.nn.Linear(32, FC[0])
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.fc_2 = torch.nn.Linear(FC[0], FC[1])
    self.bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]

    # out = x.unsqueeze(0)
    # # shape : [1, batch*N, 32]
    # out = out.view(1, out.shape[-1], -1)
    # # shape : [1, 32, batch*N]
    # out = F.relu(self.bn_1(self.conv_1d(out)))
    # out = out.squeeze(0).view(out.shape[-1], -1)
    # # shape : [batch*N, 128]
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      temp_out.append(torch.mean(x[ind_start : ind_end], 0).unsqueeze(0))

      ind_point = ind_end

    # shape : batch * [32]
    out = torch.cat(temp_out, 0)
    out = F.relu(self.bn_1(self.fc_1(out)))
    result = self.bn_2(self.fc_2(out))
    # result = self.fc_1(out)
    # shape : [batch, 128]
    # shape : [batch, 128]

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    # result = F.normalize(result, p=2)

    return result


############################################################

class FCGF_AVG2_norm(torch.nn.Module) :
  FC = [64, 128]

  def __init__(self,
               in_channels=32,
               normalize_feature=None,
               ):
    super(FCGF_AVG2_norm, self).__init__()
    FC = self.FC

    self.fc_1 = torch.nn.Linear(32, FC[0])
    self.bn_1 = torch.nn.BatchNorm1d(FC[0])
    self.fc_2 = torch.nn.Linear(FC[0], FC[1])
    self.bn_2 = torch.nn.BatchNorm1d(FC[1])

  def forward(self, x, length):
    # save index points per batch's start point
    n_per_batch = length
    # x shape : [batch*N, 32]

    # out = x.unsqueeze(0)
    # # shape : [1, batch*N, 32]
    # out = out.view(1, out.shape[-1], -1)
    # # shape : [1, 32, batch*N]
    # out = F.relu(self.bn_1(self.conv_1d(out)))
    # out = out.squeeze(0).view(out.shape[-1], -1)
    # # shape : [batch*N, 128]
    temp_out=[]
    ind_point = 0
    for i in range(len(n_per_batch)) :
      if i==0 :
        ind_start = 0
        ind_end = n_per_batch[i]
      else :
        ind_start = ind_point
        ind_end = ind_point + n_per_batch[i]

      temp_out.append(torch.mean(x[ind_start : ind_end], 0).unsqueeze(0))

      ind_point = ind_end

    # shape : batch * [32]
    out = torch.cat(temp_out, 0)
    out = F.relu(self.bn_1(self.fc_1(out)))
    result = self.bn_2(self.fc_2(out))
    # result = self.fc_1(out)
    # shape : [batch, 128]
    # shape : [batch, 128]

    '''
    마지막에 normalize를 하지 않는다면?
    '''
    result = F.normalize(result, p=2)

    return result
############################################################
