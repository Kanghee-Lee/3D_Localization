import os
import sys
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import logging

from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature_from_numpy, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results, run_ransac

import torch

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def load_single_pcfile(filename, dim=3, dtype=np.float32):
  pc = np.fromfile(filename, dtype=dtype)
  pc = np.reshape(pc, (pc.shape[0] // dim, dim))
  return pc[:, 0:3]

def quat_to_rotation(q, trans):
  T = np.eye(4)
  q = list(map(float, q))
  trans = list(map(float, trans))
  T[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
  T[0,1] = 2 * (q[1]*q[2] - q[0]*q[3])
  T[0,2] = 2 * (q[1]*q[3] + q[0]*q[2])
  T[1,0] = 2 * (q[1]*q[2] + q[0]*q[3])
  T[1,1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
  T[1,2] = 2 * (q[2]*q[3] - q[0]*q[1])
  T[2,0] = 2 * (q[1]*q[3] - q[0]*q[2])
  T[2,1] = 2 * (q[2]*q[3] + q[0]*q[1])
  T[2,2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
  T[:3, 3] = trans
  return T

def extract_features_batch(model, config, source_path, target_path, voxel_size, device):
  bin_folder = os.path.join(source_path, "oxford_test_local")
  files = get_file_list(bin_folder, ".bin")
  assert len(files) > 0, f"Could not find oxford test files under {bin_folder}"
  logging.info(files)
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for i, fi in enumerate(files):
    # Extract features from a file
    cloud = load_single_pcfile(fi)
    xyz = cloud[:, 0:3]
    pcd = make_open3d_point_cloud(xyz)
    fi_base = os.path.basename(fi)
    save_fn = fi_base
    
    if i % 100 == 0:
      logging.info(f"{i} / {len(files)}: {save_fn}")

    timer.tic()
    xyz_down, feature = extract_features(
        model,
        xyz=np.array(pcd.points),
        rgb=None,
        normal=None,
        voxel_size=voxel_size,
        device=device,
        skip_check=True)
    t = timer.toc()
    if i > 0:
      tmeter.update(t)
      num_feat += len(xyz_down)

    np.savez_compressed(
        os.path.join(target_path, save_fn),
        points=np.array(pcd.points),
        xyz=xyz_down,
        feature=feature.detach().cpu().numpy())
    if i % 20 == 0 and i > 0:
      logging.info(
          f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
      )

def registration(source, feature, voxel_size):
  """
  Gather .log files produced in --target folder and run this Matlab script
  https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
  (see Geometric Registration Benchmark section in
  http://3dmatch.cs.princeton.edu/)
  """
  with open(os.path.join(source, "oxford_test_local_gt.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  
  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
  reg_timer = Timer()

  for i in range(1,len(sets)):
    T_gt = quat_to_rotation(sets[i][7:], sets[i][4:7])
    name_i = "{}.bin".format(sets[i][0])
    name_j = "{}.bin".format(sets[i][1])

    # coord and feat form a sparse tensor.
    data_i = np.load(os.path.join(feature, name_i + ".npz"))
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
    data_j = np.load(os.path.join(feature, name_j + ".npz"))
    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

    # make pcd and feature to open3d object
    pcd0 = make_open3d_point_cloud(coord_i)
    pcd1 = make_open3d_point_cloud(coord_j)
    feat0 = make_open3d_feature_from_numpy(feat_i)
    feat1 = make_open3d_feature_from_numpy(feat_j)

    reg_timer.tic()
    distance_threshold = voxel_size * 3.0 # 1.0 ~ 3.0 
    ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
        pcd0, pcd1, feat0, feat1, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95), # 0.9 ~ 1.0
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 10000))
    T_ransac = ransac_result.transformation.astype(np.float32)
    T_ransac = torch.from_numpy(np.linalg.inv(T_ransac))
    reg_timer.toc()

    # Translation error
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gt[:3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)

    # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
    if rte < 2:
      rte_meter.update(rte)

    if not np.isnan(rre) and rre < np.pi / 180 * 5:
      rre_meter.update(rre)

    if not np.isnan(rre) and rre < np.pi / 180 * 5 and rte < 2:
      success_meter.update(1)
      logging.info(f" ({i}/{len(sets)}) Suceed with RTE: {rte}, RRE: {rre}({rre * 180 / np.pi}), Check: {2.0}, {5.0}")
    else:
      success_meter.update(0)
      logging.info(f" ({i}/{len(sets)}) Failed with RTE: {rte}, RRE: {rre}({rre * 180 / np.pi}), Check: {2.0}, {5.0}")

    if i % 10 == 0:
      logging.info(
          f" Current : {i}/{len(sets)}, " +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}({rre * 180 / np.pi}), Success: {success_meter.sum} / {success_meter.count}" +
          f" ({success_meter.avg * 100} %)")
      reg_timer.reset()
  
  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)")

def do_single_pair_evaluation(feature_path,
                              set,
                              T_gt,
                              voxel_size,
                              tau_1=0.1,
                              tau_2=0.05,
                              num_rand_keypoints=-1):
  trans_gth = np.linalg.inv(T_gt)
  name_i = "{}.bin".format(set[0])
  name_j = "{}.bin".format(set[1])

  # coord and feat form a sparse tensor.
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

  # use the keypoints in oxford
  if num_rand_keypoints > 0:
    # Randomly subsample N points
    Ni, Nj = len(points_i), len(points_j)
    inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
    inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

    sample_i, sample_j = points_i[inds_i], points_j[inds_j]

    key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))
    key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

    key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))
    key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

    inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]
    inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

    coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
    coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

  coord_i = make_open3d_point_cloud(coord_i)
  coord_j = make_open3d_point_cloud(coord_j)

  hit_ratio = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth,
                                       tau_1)

  #logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
  if hit_ratio >= tau_2:
    return True
  else:
    return False


def feature_evaluation(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
  with open(os.path.join(source_path, "oxford_test_local_gt.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]

  assert len(
      sets
  ) > 0, "No gt file is there."

  tau_1 = 0.8  # 80cm
  tau_2 = 0.04  # 4% inlier
  logging.info("%f %f" % (tau_1, tau_2))

  results = []
  for i in range(1,len(sets)):
    T_gt = quat_to_rotation(sets[i][7:], sets[i][4:7])
    results.append(
        do_single_pair_evaluation(feature_path, sets[i], T_gt, voxel_size, tau_1,
                                  tau_2, num_rand_keypoints))
    if (i % 100 == 0):
      print("{}/{}".format(i, len(sets)))

  mean_recall = np.array(results).mean()
  std_recall = np.array(results).std()
  logging.info(f'result: {mean_recall} +- {std_recall}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to oxford test dataset')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
      '-m',
      '--model',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.20,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', action='store_true')
  parser.add_argument('--evaluate_feature_match_recall', action='store_true')
  parser.add_argument(
      '--evaluate_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument('--with_cuda', action='store_true')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=15000,
      help='Number of random keypoints for each scene')
  args = parser.parse_args()

  device = torch.device('cuda' if args.with_cuda else 'cpu')

  if args.extract_features:
    assert args.model is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    with torch.no_grad():
      extract_features_batch(model, config, args.source, args.target, config.voxel_size,
                             device)
  
  if args.evaluate_feature_match_recall:
    assert (args.target is not None)
    with torch.no_grad():
      feature_evaluation(args.source, args.target, args.voxel_size,
                         args.num_rand_keypoints)
  
  if args.evaluate_registration:
    assert (args.source is not None)
    with torch.no_grad():
      registration(args.source, args.target, args.voxel_size)