# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib
import pickle

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
from util.augment import get_augmentations_from_list
import lib.transforms as t

import MinkowskiEngine as ME

import open3d as o3d


def collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
      zip(*list_data))
  xyz_batch0, xyz_batch1 = [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))

  def to_tensor(x):
    if isinstance(x, torch.Tensor):
      return x
    elif isinstance(x, np.ndarray):
      return torch.from_numpy(x)
    else:
      raise ValueError(f'Can not convert to torch tensor, {x}')

  for batch_id, _ in enumerate(coords0):
    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    xyz_batch0.append(to_tensor(xyz0[batch_id]))
    xyz_batch1.append(to_tensor(xyz1[batch_id]))

    trans_batch.append(to_tensor(trans[batch_id]))

    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
  coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0.float(),
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1.float(),
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch
  }


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  # R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  # upper is 3-axsis rotation
  R = np.random.uniform(low=-np.pi, high=np.pi)
  cosval = np.cos(R)
  sinval = np.sin(R)
  R = np.array([[cosval, sinval, 0],
                              [-sinval, cosval, 0],
                              [0, 0, 1]])
  T[:3, :3] = R
  #T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T

def load_single_pcfile(filename, dim=3, dtype=np.float32):
    pc = np.fromfile(filename, dtype=dtype)
    pc = np.reshape(pc, (pc.shape[0] // dim, dim))
    return pc[:, 0:3]

class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)

class OxfordTrainDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.oxford_dir
    logging.info(f"Loading the subset {phase} from {root}")

    self.augmentation = get_augmentations_from_list(['Jitter'], upright_axis=2)
    train_file = os.path.join(root, 'oxford_train_local_gt.pickle')
    with open(train_file, 'rb') as handle:
      self.files = pickle.load(handle)
    if phase == 'train':
      self.start_ = 0
      self.end_ = len(self.files.keys()) - 2400
    elif phase == 'val':
      self.start_ = len(self.files.keys()) - 2400
      self.end_ = len(self.files.keys())
    
  def process_point_cloud(self, cloud):
        # augmentation
        for a in self.augmentation:
            cloud = a.apply(cloud)
        return cloud

  def __getitem__(self, idx):
    pcfile = self.files[self.start_ + idx]['query']
    pcfile = os.path.join(self.root, pcfile+'.bin')
    cloud = load_single_pcfile(pcfile, dim=3)
    xyz0 = self.process_point_cloud(cloud[:, 0:3])
    xyz1 = self.process_point_cloud(cloud[:, 0:3])
    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
      '''
      rotation_angle = np.random.uniform(low=-np.pi, high=self.np.pi)
      cosval = np.cos(rotation_angle)
      sinval = np.sin(rotation_angle)
      trans = np.array([[cosval, sinval, 0],
                                  [-sinval, cosval, 0],
                                  [0, 0, 1]])
      xyz1 = np.dot(xyz1, trans)
      '''
    else:
      trans = np.identity(4)

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(pcd0.points)
    npts1 = len(pcd1.points)

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)
  
  def __len__(self):
    return self.end_ - self.start_


ALL_DATASETS = [OxfordTrainDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  assert phase in ['train', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  #if phase in ['train']:
  use_random_rotation = config.use_random_rotation
  use_random_scale = config.use_random_scale
  transforms += [t.Jitter()]

  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config)

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      drop_last=True)

  return loader
