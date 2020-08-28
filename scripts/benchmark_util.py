import open3d as o3d
import os
import logging
import numpy as np

from util.trajectory import CameraPose
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy

from scipy import spatial

def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
      xyz0, xyz1, feat0, feat1, distance_threshold,
      o3d.registration.TransformationEstimationPointToPoint(False), 4, [
          o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
  return result_ransac.transformation


def gather_results(results):
  traj = []
  for r in results:
    success = r[0]
    if success:
      traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
  return traj


def gen_matching_pair(pts_num):
  matching_pairs = []
  for i in range(pts_num):
    for j in range(i + 1, pts_num):
      matching_pairs.append([i, j, pts_num])
  return matching_pairs


def read_data(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  xyz = make_open3d_point_cloud(data['xyz'])
  feat = make_open3d_feature_from_numpy(data['feature'])
  print("In benchmark_util.py - data['feature'] : ", data['feature'])
  print("In benchmark_util.py - data['feature'] shape : ", data['feature'].shape)
  print("In benchmark_util.py - feat : ", feat)

  print('-'*20)
  return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  logging.info("matching %s %s" % (name_i, name_j))
  print("feature_path : ", feature_path)
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)
  print("points, xyz, feats : ", points_j, xyz_j, feat_j)
  if len(xyz_i.points) < len(xyz_j.points):
    trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  else:
    trans = run_ransac(xyz_j, xyz_i, feat_j, feat_i, voxel_size)
    trans = np.linalg.inv(trans)
  ratio = compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
  logging.info(f"{ratio}")
  if ratio > 0.3:
    return [True, i, j, s, np.linalg.inv(trans)]
  else:
    return [False, i, j, s, np.identity(4)]

#######################################################################


def global_desc_matching(feature_path, set_name, m, voxel_size, pooling_arg) :
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  logging.info("matching %s %s" % (name_i, name_j))
#  print("feature_path : ", feature_path)
  feat_np_i = read_dataAsnumpy(feature_path, name_i)
  feat_np_j = read_dataAsnumpy(feature_path, name_j)

  feat_np_i = feat_pooling(feat_np_i, pooling_arg)
  feat_np_j = feat_pooling(feat_np_j, pooling_arg)

  cos_sim = 1 - spatial.distance.cosine(feat_np_i, feat_np_j)
#  print('feature & cos_sim')
#  print(cos_sim)

  if cos_sim > 0.5 :
    return [True, i, j, s, cos_sim]
  else : 
    return [False, i, j, s, cos_sim]

def read_dataAsnumpy(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  return data['feature']

def feat_pooling(feat, pooling_arg) :
  if pooling_arg == 'max' :
    print('------------Max--------------')
    print(feat)
    print(np.max(feat, 0))
    return np.max(feat, 0)
  elif pooling_arg == 'avg' :
    return np.average(feat, 0) 
  
