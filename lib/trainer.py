# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
from scipy.spatial import cKDTree
import json
from torchviz import make_dot
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

import MinkowskiEngine as ME


class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
      freeze=False
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    if config.load_fcgf :
      Model = load_model(config.model)
      model = Model()
    else :
      Model = load_model(config.model)
      model = Model(
          num_feats,
          config.model_n_out,
          bn_momentum=config.bn_momentum,
          normalize_feature=config.normalize_feature,
          conv1_kernel_size=config.conv1_kernel_size,
          D=3)

    if config.weights and not config.load_fcgf:
      # 수정
      # pretrained_dict = ...
      # model_dict = model.state_dict()
      #
      # # 1. filter out unnecessary keys
      # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      # # 2. overwrite entries in the existing state dict
      # model_dict.update(pretrained_dict)
      # # 3. load the new state dict
      # model.load_state_dict(pretrained_dict)
      # print('1'*20)
      checkpoint = torch.load(config.weights)
      # print(checkpoint['config'])
      # print(config)
      pretrained_dict = checkpoint['state_dict']
      model_dict = model.state_dict()
      # print('2' * 20)
      # print(pretrained_dict['conv1.kernel'].shape)
      # print(model_dict['conv1.kernel'].shape)
      # print(model_dict.keys())
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      # print('3' * 20)
      model.load_state_dict(model_dict, strict=False)
      # print('4' * 20)
      # print('8'*20)
      # print(checkpoint)

      #model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)
    # 수정
    self.freeze = freeze
    self.load_fcgf = config.load_fcgf

    self.target_path = config.target_path
    self.config = config

    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = np.inf
    self.checkpoint_name = config.checkpoint_name
    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 수정
    # if freeze and not self.load_fcgf:
    #   for name, p in model.named_parameters():
    #     if 'conv_1d' not in name or 'bn_1' not in name:
    #       p.requires_grad = False
      # for name, p in model.named_parameters() :
      #   print(name, p.requires_grad)

    params = [p for p in model.parameters() if p.requires_grad]
    # print(params)
    self.optimizer = getattr(optim, config.optimizer)(
        params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)
    # 수정
    self.model = model

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    # self.model = torch.nn.DataParallel(self.model)
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """

    # # 수정
    # if freeze :
    #   with torch.no_grad() :
    #     for name, p in self.model.named_parameters() :
    #       if 'conv_1d' in name or 'bn_1' in name :
    #         dd

    # ver5

    print('------------Training start------------')
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch, self.checkpoint_name)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val > val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

####################################################################################################################

  def train_mlp(self):
    """
    Full training logic
    """

    # # 수정
    # if freeze :
    #   with torch.no_grad() :
    #     for name, p in self.model.named_parameters() :
    #       if 'conv_1d' in name or 'bn_1' in name :
    #         dd

    # ver5

    print('------------Training start------------')
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch_mlp()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch_mlp(epoch)
      self._save_checkpoint(epoch, self.checkpoint_name)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch_mlp()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val > val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, self.checkpoint_name + 'best')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

####################################################################################################################
  def train_reg(self):
    """
    Full training logic
    """

    # # 수정
    # if freeze :
    #   with torch.no_grad() :
    #     for name, p in self.model.named_parameters() :
    #       if 'conv_1d' in name or 'bn_1' in name :
    #         dd

    # ver5

    print('------------Training start------------')
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch_reg()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch_reg(epoch)
      self._save_checkpoint(epoch, self.checkpoint_name)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch_reg()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val > val_dict[self.best_val_metric]:
          logging.info(
            f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, self.checkpoint_name + 'best')
        else:
          logging.info(
            f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

####################################################################################################################


  def _save_checkpoint(self, epoch, filename='checkpoint_128_th1.4'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

  # # ver5
  # def _save_fcgf(self, epoch, filename='checkpoint'):
  #   # state = {
  #   #     'epoch': epoch,
  #   #     'state_dict': self.model.state_dict(),
  #   #     'optimizer': self.optimizer.state_dict(),
  #   #     'scheduler': self.scheduler.state_dict(),
  #   #     'config': self.config,
  #   #     'best_val': self.best_val,
  #   #     'best_val_epoch': self.best_val_epoch,
  #   #     'best_val_metric': self.best_val_metric
  #   # }
  #   np.savez_compressed(
  #     os.path.join(target_path, save_fn),
  #     points=np.array(pcd.points),
  #     xyz=xyz_down,
  #     feature=feature.detach().cpu().numpy())
  #   filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
  #   logging.info("Saving checkpoint: {} ...".format(filename))
  #   torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
      freeze=False,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight
    self.gp_weight = config.gp_weight
  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#        F0 = self.model(sinput0).F
        F0 = self.model(sinput0)
        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
#        F1 = self.model(sinput1).F
        F1 = self.model(sinput1)
        N0, N1 = len(sinput0), len(sinput1)

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#      F0 = self.model(sinput0).F
      F0 = self.model(sinput0)
      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
#      F1 = self.model(sinput1).F
      F1 = self.model(sinput1)
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

#####################################################################

class ContrastiveLossTrainer_MLP(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
      freeze=False,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight
    self.gp_weight = config.gp_weight
  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  # ver5
  def _extract(self):
    gc.collect()
    self.model.eval()

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = 0

    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    val_data_loader_iter = self.val_data_loader.__iter__()
    print('trainer.py - extract - tot_num : ', tot_num_data)
    for batch_idx in range(tot_num_data):
      val_input_dict = val_data_loader_iter.next()
      sinput0 = ME.SparseTensor(
        val_input_dict['sinput0_F'], coords=val_input_dict['sinput0_C']).to(self.device)
      F0,F1, F2, F3, fcgf_length0 = self.model(sinput0)
      np.savez_compressed(
        os.path.join(self.target_path, val_input_dict['file0'][0]),
        fcgf0=F0.detach().cpu().numpy(),
        fcgf1=F1.detach().cpu().numpy(),
        fcgf2=F2.detach().cpu().numpy(),
        fcgf3=F3.detach().cpu().numpy(),
        length=fcgf_length0)

    for curr_iter in range(len(data_loader) // iter_size):
      for iter_idx in range(iter_size):
        input_dict = data_loader_iter.next()
        sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        F0, F1, F2, F3, fcgf_length0 = self.model(sinput0)
        np.savez_compressed(
          os.path.join(self.target_path, input_dict['file0'][0]),
          fcgf0=F0.detach().cpu().numpy(),
          fcgf1=F1.detach().cpu().numpy(),
          fcgf2=F2.detach().cpu().numpy(),
          fcgf3=F3.detach().cpu().numpy(),
          length = fcgf_length0)

      if curr_iter % self.config.stat_freq == 0:
        logging.info(
          "FCGF extract Epoch: [{}/{}]"
          .format(curr_iter,
                  len(self.data_loader) //
                  iter_size, ))



  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, model_meter, train_meter, data_timer, total_timer, model_timer, train_timer = AverageMeter(), AverageMeter(), AverageMeter(), Timer(), Timer(), Timer(), Timer()

    # Main training

    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      model_time = 0
      train_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        model_timer.tic()
        # pairs consist of (xyz1 index, xyz0 index)
        # print('in'*20)
        # print('train_epoch - input_dict.shape : ', type(input_dict['sinput0_C']))
        # print('train_epoch - input_dict.shape2 : ', input_dict['sinput0_F'].shape)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#        F0 = self.model(sinput0).F
        F0 = self.model(sinput0)
        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
#        F1 = self.model(sinput1).F
        F1 = self.model(sinput1)

        model_time += model_timer.toc(average=False)
        # print('trainer.py - F0 shape', F0.shape)
        N0, N1 = len(sinput0), len(sinput1)
        # print(len(data_loader))
        # print(len(input_dict))
        # 수정

        # pos_pairs = input_dict['correspondences']
        # neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        # pos_pairs = pos_pairs.long().to(self.device)
        # neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)
        #
        # neg0 = F0.index_select(0, neg_pairs[:, 0])
        # neg1 = F1.index_select(0, neg_pairs[:, 1])
        # pos0 = F0.index_select(0, pos_pairs[:, 0])
        # pos1 = F1.index_select(0, pos_pairs[:, 1])

        # print('trainer.py - train_epoch : ', input_dict['pcd_match'])
        # print(F0.shape)
        # print(F1.shape)
        train_timer.tic()
        pos_ind=[]
        neg_ind=[]
        # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
        for ind in range(len(input_dict['pcd_match'])) :
          if input_dict['pcd_match'][ind] :
            pos_ind.append(ind)
          else :
            neg_ind.append(ind)
        if pos_ind :
          pos_loss = ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))
        else :
          pos_loss = torch.tensor(0.)
        if neg_ind :
          neg_loss = (F.relu(self.neg_thresh -
                                  ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))
        else :
          neg_loss = torch.tensor(0.)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        loss = pos_loss_mean + self.neg_weight * neg_loss_mean

        # if input_dict['pcd_match'] :
        #   pos_loss_mean = ((F0 - F1).pow(2).sum(1)).mean() / iter_size
        #
        #   neg_loss_mean = 0
        #   loss = pos_loss_mean
        #   pos_loss_mean = pos_loss_mean.item()
        #   # print('------------------Positive------------------')
        #   # print('trainer.py')
        #   # print('F0', F0)
        #   # print('F1', F0)
        #   # print('pos_loss', loss.item())
        #
        # else :
        #   pos_loss_mean = 0
        #   neg_loss_mean = (F.relu(self.neg_thresh -
        #                   ((F0 - F1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)).mean() / iter_size
        #
        #   loss = neg_loss_mean
        #   neg_loss_mean = neg_loss_mean.item()
        #   # print('------------------Negative------------------')
        #   # print('trainer.py')
        #   # print('F0', F0)
        #   # print('F1', F0)
        #   # print('neg_loss', loss.item())


        # # Positive loss
        # pos_loss = (F0 - F1).pow(2).sum(1)
        #
        # # Negative loss
        # neg_loss = F.relu(self.neg_thresh -
        #                   ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)
        #
        # pos_loss_mean = pos_loss.mean() / iter_size
        # neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        # loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the beginning of iter_size
        # batch_loss += loss.item()
        # batch_pos_loss += pos_loss_mean.item()
        # batch_neg_loss += neg_loss_mean.item()
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

        train_time += train_timer.toc(average=False)
      self.optimizer.step()

      torch.cuda.empty_cache()



      total_timer.toc()
      data_meter.update(data_time)
      model_meter.update(model_time)
      train_meter.update(train_time)
      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)

        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, Model time: {:.4f}, Trainer time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, model_meter.avg, train_meter.avg))
        data_meter.reset()
        total_timer.reset()

########################################################################################################################

  def _train_epoch_mlp(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, model_meter, train_meter, data_timer, total_timer, model_timer, train_timer = AverageMeter(), AverageMeter(), AverageMeter(), Timer(), Timer(), Timer(), Timer()

    # Main training

    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      model_time = 0
      train_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        model_timer.tic()

        fcgf0 = input_dict['fcgf0']
        fcgf1 = input_dict['fcgf1']
        length0 = input_dict['length0']
        length1 = input_dict['length1']
        overlap = np.array(input_dict['overlap'])
        overlap = torch.from_numpy(overlap).to(self.device)
        fcgf0, fcgf1 = fcgf0.to(self.device), fcgf1.to(self.device)

        F0 = self.model(fcgf0, length0)
        F1 = self.model(fcgf1, length1)

        model_time += model_timer.toc(average=False)
        train_timer.tic()
        pos_ind = []
        neg_ind = []
        # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
        for ind in range(len(input_dict['pcd_match'])):
          if input_dict['pcd_match'][ind]:
            pos_ind.append(ind)
          else:
            neg_ind.append(ind)
        pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
        # print(overlap)
        # print(overlap[pos_ind])
        # assert 0
        if pos_ind:
          pos_loss = (F.relu(
                             ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1) + 1e-7).sqrt()-(1-overlap[pos_ind])).pow(2))
          # pos_loss = ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))

        if neg_ind:
          neg_loss = (F.relu(self.neg_thresh -
                             ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-7).sqrt()).pow(2))

        # make_dot(F0, params=dict(list(self.model.named_parameters()))).render("att_k_fc", format="png")
        # assert 0
        # print((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))
        # print(((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1) + 1e-7).sqrt())
        # print(F.relu(((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1) + 1e-7).sqrt()-(1-overlap[pos_ind])).pow(2))
        # print(overlap[pos_ind])
        # print('pos : ', ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1) + 1e-7).sqrt())
        # print('neg : ', ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-7).sqrt())
        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        # Weighted loss
        # loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the beginning of iter_size

        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

        train_time += train_timer.toc(average=False)

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_timer.toc()
      data_meter.update(data_time)
      model_meter.update(model_time)
      train_meter.update(train_time)
      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)

        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, Model time: {:.4f}, Trainer time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, model_meter.avg, train_meter.avg))
        data_meter.reset()
        total_timer.reset()

########################################################################################################################
  def _train_epoch_reg(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, model_meter, train_meter, data_timer, total_timer, model_timer, train_timer = AverageMeter(), AverageMeter(), AverageMeter(), Timer(), Timer(), Timer(), Timer()

    # Main training

    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss, batch_gp_loss = 0, 0, 0, 0

      data_time = 0
      model_time = 0
      train_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        model_timer.tic()

        # sinput0 = ME.SparseTensor(
        #   input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        # #        F0 = self.model(sinput0).F
        # F0 = self.model(sinput0)
        # sinput1 = ME.SparseTensor(
        #   input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
        # #        F1 = self.model(sinput1).F
        # F1 = self.model(sinput1)

        fcgf0 = input_dict['fcgf0']
        fcgf1 = input_dict['fcgf1']
        length0 = input_dict['length0']
        length1 = input_dict['length1']

        fcgf0, fcgf1 = fcgf0.to(self.device), fcgf1.to(self.device)

        F0, gp0, n0 = self.model(fcgf0, length0)
        F1, gp1, n1 = self.model(fcgf1, length1)
        n0 = torch.from_numpy(n0).to(self.device)
        n1 = torch.from_numpy(n1).to(self.device)

        gp_loss0= torch.norm((gp0-(n0//8)), dim=(1))
        gp_loss0 = gp_loss0.mean()

        gp_loss1 = torch.norm((gp1 - (n1 // 8)), dim=(1))
        gp_loss1 = gp_loss1.mean()

        gp_loss = gp_loss0 + gp_loss1

        model_time += model_timer.toc(average=False)
        train_timer.tic()
        pos_ind = []
        neg_ind = []
        # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
        for ind in range(len(input_dict['pcd_match'])):
          if input_dict['pcd_match'][ind]:
            pos_ind.append(ind)
          else:
            neg_ind.append(ind)
        pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
        if pos_ind:
          pos_loss = ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))

        if neg_ind:
          neg_loss = (F.relu(self.neg_thresh -
                             ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))

        # make_dot(F0, params=dict(list(self.model.named_parameters()))).render("att_k_fc", format="png")
        # assert 0

        # for name, param in self.model.named_parameters():
        #   print(name, param.requires_grad)
        #   print(param.data)
        # print(F0[0])
        #
        # print(F1[0])

        # if True in input_dict['pcd_match'] :
        #
        #   print(input_dict['file0'], input_dict['file1'], input_dict['pcd_match'])
        #   assert 0
        # print(input_dict['pcd_match'])
        # print(pos_loss)
        # print('-'*30)
        # print((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1))
        # print(self.neg_thresh-((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1)+1e-4).sqrt())
        # print((self.neg_thresh - ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))
        # print(F.relu(self.neg_thresh - ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))
        # print(neg_loss)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean + self.gp_weight * gp_loss
        # Weighted loss
        # loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the beginning of iter_size

        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()
        batch_gp_loss += gp_loss.item()
        train_time += train_timer.toc(average=False)
      self.optimizer.step()

      torch.cuda.empty_cache()

      total_timer.toc()
      data_meter.update(data_time)
      model_meter.update(model_time)
      train_meter.update(train_time)
      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/gp_loss', batch_gp_loss, start_iter + curr_iter)

        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f} Gp: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss, batch_gp_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, Model time: {:.4f}, Trainer time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, model_meter.avg, train_meter.avg))
        data_meter.reset()
        total_timer.reset()

########################################################################################################################

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()
    loss=0
    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#      F0 = self.model(sinput0).F
      F0 = self.model(sinput0)
      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
#      F1 = self.model(sinput1).F
      F1 = self.model(sinput1)
      feat_timer.toc()

      pos_ind = []
      neg_ind = []
      # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
      for ind in range(len(input_dict['pcd_match'])):
        if input_dict['pcd_match'][ind]:
          pos_ind.append(ind)
        else:
          neg_ind.append(ind)
      pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
      if pos_ind:
        pos_loss = ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))

      if neg_ind:
        neg_loss = (F.relu(self.neg_thresh -
                           ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))

      pos_loss_mean = pos_loss.mean()
      neg_loss_mean = neg_loss.mean()
      temp_loss = pos_loss_mean + self.neg_weight * neg_loss_mean

      loss += temp_loss.item()



      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Loss: {(loss/num_data):.3f},"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {(loss/num_data):.3f},"
    ]))
    return {
        "final_loss": (loss/num_data),
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

########################################################################################################################

  def _valid_epoch_mlp(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()
    loss = 0
    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()

      fcgf0 = input_dict['fcgf0']
      fcgf1 = input_dict['fcgf1']
      length0 = input_dict['length0']
      length1 = input_dict['length1']
      overlap = np.array(input_dict['overlap'])
      overlap = torch.from_numpy(overlap).to(self.device)
      fcgf0, fcgf1 = fcgf0.to(self.device), fcgf1.to(self.device)

      F0 = self.model(fcgf0, length0)
      F1 = self.model(fcgf1, length1)

      feat_timer.toc()
      pos_ind = []
      neg_ind = []
      # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
      for ind in range(len(input_dict['pcd_match'])):
        if input_dict['pcd_match'][ind]:
          pos_ind.append(ind)
        else:
          neg_ind.append(ind)
      pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
      if pos_ind:
        pos_loss = (F.relu(
          ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1) + 1e-7).sqrt() - (1 - overlap[pos_ind])).pow(2))

      if neg_ind:
        neg_loss = (F.relu(self.neg_thresh -
                           ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-7).sqrt()).pow(2))

      pos_loss_mean = pos_loss.mean()
      neg_loss_mean = neg_loss.mean()
      temp_loss = pos_loss_mean + self.neg_weight * neg_loss_mean
      loss += temp_loss.item()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Loss: {(loss / num_data):.3f},"
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {(loss / num_data):.3f},"
    ]))
    return {
      "final_loss": (loss / num_data),
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg
    }

########################################################################################################################

  def _valid_epoch_reg(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()
    loss = 0
    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()

      fcgf0 = input_dict['fcgf0']
      fcgf1 = input_dict['fcgf1']
      length0 = input_dict['length0']
      length1 = input_dict['length1']

      fcgf0, fcgf1 = fcgf0.to(self.device), fcgf1.to(self.device)


      F0, gp0, n0 = self.model(fcgf0, length0)
      F1, gp1, n1 = self.model(fcgf1, length1)
      n0 = torch.from_numpy(n0).to(self.device)
      n1 = torch.from_numpy(n1).to(self.device)
      gp_loss0 = torch.norm((gp0 - (n0 // 8)), dim=(1))
      gp_loss0 = gp_loss0.mean()

      gp_loss1 = torch.norm((gp1 - (n1 // 8)), dim=(1))
      gp_loss1 = gp_loss1.mean()

      gp_loss = gp_loss0 + gp_loss1

      feat_timer.toc()
      pos_ind = []
      neg_ind = []
      # pos_ind = [ind for ind in range(len(input_dict['pcd_match'])) if input_dict['pcd_match'][ind] == True]
      for ind in range(len(input_dict['pcd_match'])):
        if input_dict['pcd_match'][ind]:
          pos_ind.append(ind)
        else:
          neg_ind.append(ind)
      pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
      if pos_ind:
        pos_loss = ((F0[pos_ind] - F1[pos_ind]).pow(2).sum(1))

      if neg_ind:
        neg_loss = (F.relu(self.neg_thresh -
                           ((F0[neg_ind] - F1[neg_ind]).pow(2).sum(1) + 1e-4).sqrt()).pow(2))

      pos_loss_mean = pos_loss.mean()
      neg_loss_mean = neg_loss.mean()
      temp_loss = pos_loss_mean + self.neg_weight * neg_loss_mean + self.gp_weight * gp_loss
      loss += temp_loss.item()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Loss: {(loss / num_data):.3f},"
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {(loss / num_data):.3f},"
    ]))
    return {
      "final_loss": (loss / num_data),
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg
    }

########################################################################################################################
  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

#####################################################################


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """

################################################################
    # print('='*50)
    # print(positive_pairs)

    N0, N1 = len(F0), len(F1)

    # print('trainer.py - contrastive_ - N0, N1 : ', N0, N1)
    # print('trainer.py - contrastive_ - F0 shape : ', F0.shape)


    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#        F0 = self.model(sinput0).F
        F0 = self.model(sinput0)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

#        F1 = self.model(sinput1).F
        F1 = self.model(sinput1)
        # print('-'*50)
        # print('trainer.py - input_dict : ', input_dict)
        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()
#################################################################################################
########################################################################################################################

  def _train_epoch_mlp(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, model_meter, train_meter, data_timer, total_timer, model_timer, train_timer = AverageMeter(), AverageMeter(), AverageMeter(), Timer(), Timer(), Timer(), Timer()

    # Main training

    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      model_time = 0
      train_time = 0
      total_timer.tic()
      train_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        model_timer.tic()
        fcgf = input_dict['fcgf']

        length = input_dict['length']
        overlap = input_dict['overlap']
        pcd_match = input_dict['pcd_match']

        overlap = torch.from_numpy(overlap).to(self.device)
        fcgf = fcgf.to(self.device)

        pcd_match = torch.from_numpy(pcd_match).to(self.device)
        F0, gp0, n0 = self.model(fcgf, length)
        n0 = torch.from_numpy(n0).to(self.device)

        gp_loss0 = torch.norm((gp0 - (n0 // 8)), dim=(1))

        gp_loss0 = gp_loss0.mean()

        gp_loss = gp_loss0

        if F0.shape[0] ==2 :
          neg_loss = torch.tensor(0.)
          pos_D = pdist(F0[0:1], F0[1:], dist_type='L2')
          pos_loss = F.relu((pos_D-(1-overlap))).pow(2)
          # pos_loss = pos_D.pow(2)
        else :
          pos_D = pdist(F0[0:1], F0[1:2], dist_type='L2')
          neg_D = pdist(F0[0:1], F0[2:], dist_type='L2')
          Dmin, Dind = neg_D.min(1)
          pos_loss = F.relu((pos_D - (1 - overlap))).pow(2)
          # pos_loss = pos_D.pow(2)
          neg_loss = F.relu(self.neg_thresh - Dmin).pow(2)

        pos_loss_mean = pos_loss.mean()
        neg_loss_mean = neg_loss.mean()

        ## hardest mining (batch 10)을 위해 삭제
        # fcgf = input_dict['fcgf']
        # length = input_dict['length']
        # overlap = input_dict['overlap']
        # pcd_match = input_dict['pcd_match']
        #
        #
        # pos_pcd_match = pcd_match.copy()
        # neg_pcd_match = pcd_match.copy()
        # neg_pcd_match = np.logical_not(neg_pcd_match)
        #
        # for i in range(len(pcd_match)) :
        #   pos_pcd_match[i][i] = False
        #   neg_pcd_match[i][i] = False
        #
        #
        # overlap = torch.from_numpy(overlap).to(self.device)
        # fcgf = fcgf.to(self.device)
        #
        # pos_pcd_match = torch.from_numpy(pos_pcd_match).to(self.device)
        # neg_pcd_match = torch.from_numpy(neg_pcd_match).to(self.device)
        # F0 = self.model(fcgf, length)
        # D = pdist(F0[0], F0[1:], dist_type='L2')
        #
        # mask = (torch.ones((len(pcd_match), len(pcd_match))) * 1e+7).to(self.device)
        # pos_mask = torch.logical_not(torch.triu(neg_pcd_match)) * mask
        #
        # pos_D = D[torch.triu(pos_pcd_match)]
        # pos_overlap = overlap[torch.triu(pos_pcd_match)]
        #
        # neg_D = D * neg_pcd_match + pos_mask
        #
        # Dmin, Dind = neg_D.min(1)
        # Dmin = Dmin[torch.where(Dmin != 1e+7)[0]]
        #
        # pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
        # if pos_pcd_match.any() :
        #   pos_loss = F.relu((pos_D-(1-pos_overlap))).pow(2)
        #
        # if neg_pcd_match.any() :
        #   neg_loss = F.relu(self.neg_thresh - Dmin).pow(2)
        #
        # pos_loss_mean = pos_loss.mean()
        # neg_loss_mean = neg_loss.mean()
        ## hardest mining (batch 10)을 위해 삭제


        loss = 0.2*pos_loss_mean + self.neg_weight * neg_loss_mean + self.gp_weight * gp_loss

        loss.backward(
        )  # To accumulate gradient, zero gradients only at the beginning of iter_size

        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

        train_time += train_timer.toc(average=False)
      self.optimizer.step()

      torch.cuda.empty_cache()

      total_timer.toc()
      data_meter.update(data_time)
      model_meter.update(model_time)
      train_meter.update(train_time)
      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)

        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, Model time: {:.4f}, Trainer time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, model_meter.avg, train_meter.avg))
        data_meter.reset()
        total_timer.reset()

########################################################################################################################

  def _valid_epoch_mlp(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()
    loss = 0

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()

      fcgf = input_dict['fcgf']
      length = input_dict['length']
      overlap = input_dict['overlap']
      pcd_match = input_dict['pcd_match']

      overlap = torch.from_numpy(overlap).to(self.device)
      fcgf = fcgf.to(self.device)
      pcd_match = torch.from_numpy(pcd_match).to(self.device)
      # F0 = self.model(fcgf, length)
      ## regulariazation)
      F0, gp0, n0 = self.model(fcgf, length)
      n0 = torch.from_numpy(n0).to(self.device)

      gp_loss0 = torch.norm((gp0 - (n0 // 8)), dim=(1))

      gp_loss0 = gp_loss0.mean()

      gp_loss = gp_loss0
      if F0.shape[0] == 2:
        neg_loss = torch.tensor(0.)
        pos_D = pdist(F0[0:1], F0[1:], dist_type='L2')
        pos_loss = F.relu((pos_D - (1 - overlap))).pow(2)
        # pos_loss = pos_D.pow(2)
      else:
        pos_D = pdist(F0[0:1], F0[1:2], dist_type='L2')
        neg_D = pdist(F0[0:1], F0[2:], dist_type='L2')
        Dmin, Dind = neg_D.min(1)
        pos_loss = F.relu((pos_D - (1 - overlap))).pow(2)
        # pos_loss = pos_D.pow(2)
        neg_loss = F.relu(self.neg_thresh - Dmin).pow(2)

      pos_loss_mean = pos_loss.mean()
      neg_loss_mean = neg_loss.mean()

      temp_loss = pos_loss_mean + self.neg_weight * neg_loss_mean+ self.gp_weight * gp_loss
      loss += temp_loss.item()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Loss: {(loss / num_data):.3f},"
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {(loss / num_data):.3f},"
    ]))
    return {
      "final_loss": (loss / num_data),
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg
    }

########################################################################################################################




class HardestContrastiveLossTrainer_MLP(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    '''
    F0, F1 : 바로 밑에 train_epoch 함수에서 data_loader_iter.netx()를 통해 생긴 input_dict에서 coord, feature를 받아와서
    모델을 통과시켜 나온 FCGF가 F0, F1이다.
    positive_pairs : 마찬가지로 train_epoch 함수에서 input_dict['correspondences'] 를 통해 가져오는데 이는 data_loaders.py에서
    collate_pair_fn에서 matching_inds에 해당됨. get_matching_indices에서는 KDTree등을 이용해 일일히 data에서 matching positive pair를 구함
    collate_fn : batch로 묶어주는 역할
    ----------------------------------------
    
    '''
################################################################

    # print('='*50)
    # print(positive_pairs)

    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#        F0 = self.model(sinput0).F
        F0 = self.model(sinput0)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

#        F1 = self.model(sinput1).F
        F1 = self.model(sinput1)

        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()


class TripletLossTrainer(ContrastiveLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=None,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

    return loss, pos_dist.mean(), rand_neg_dist.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
#        F0 = self.model(sinput0).F
        F0 = self.model(sinput0)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
#        F1 = self.model(sinput1).F
        F1 = self.model(sinput1)

        pos_pairs = input_dict['correspondences']
        loss, pos_dist, neg_dist = self.triplet_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=config.triplet_num_pos * config.batch_size,
            num_hn_samples=config.triplet_num_hn * config.batch_size,
            num_rand_triplet=config.triplet_num_rand * config.batch_size)
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        pos_dist_meter.reset()
        neg_dist_meter.reset()
        data_meter.reset()
        total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=512,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(
        torch.cat([
            rand_pos_dist + self.neg_thresh - rand_neg_dist,
            pos_dist[mask0] + self.neg_thresh - D01min[mask0],
            pos_dist[mask1] + self.neg_thresh - D10min[mask1]
        ])).mean()

    return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2
