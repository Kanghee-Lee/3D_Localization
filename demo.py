import os
import numpy as np
import argparse
import open3d as o3d
from util.visualization import get_colored_point_cloud_feature, get_colored_point_cloud_feature_pair
from util.pointcloud import make_open3d_point_cloud
from util.misc import extract_features

from model.resunet import ResUNetBN2C

import torch
import pickle

def quat_to_rotation(q, trans):
  T = np.eye(4)
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

def load_single_pcfile(filename, dim=3, dtype=np.float32):
    pc = np.fromfile(filename, dtype=dtype)
    pc = np.reshape(pc, (pc.shape[0] // dim, dim))
    return pc[:, 0:3]

def demo(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load(config.model)
  model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=5, D=3)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  model = model.to(device)

  for i in range(0, 1):
    '''
    train_file = os.path.join(config.input, 'oxford_train_local_gt.pickle')
    files = []
    with open(train_file, 'rb') as handle:
      files = pickle.load(handle)

    pcfile = files[i]['query']
    pcfile = os.path.join(config.input, pcfile+'.bin')
    '''
    # make color map by tsne for a pair of pcds.
    pcfile1 = "./testdata/oxford/oxford_test_local/642.bin"
    pcfile2 = "./testdata/oxford/oxford_test_local/268.bin"

    gt = quat_to_rotation([0.989177762028298, -0.00261356451478497, 0.0257032577246638, 0.144429453130425],
                          [0.137435455042014, -0.304622700664087, -0.059227424489])

    cloud = load_single_pcfile(pcfile1, dim=3)
    xyz = cloud[:, 0:3]
    pcd1 = make_open3d_point_cloud(xyz)
    cloud = load_single_pcfile(pcfile2, dim=3)
    xyz = cloud[:, 0:3]
    pcd2 = make_open3d_point_cloud(xyz)
    pcd2.transform(gt)

    xyz_down1, feature1 = extract_features(
        model,
        xyz=np.array(pcd1.points),
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)

    vis_pcd1 = o3d.geometry.PointCloud()
    vis_pcd1.points = o3d.utility.Vector3dVector(xyz_down1)

    xyz_down2, feature2 = extract_features(
        model,
        xyz=np.array(pcd2.points),
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)

    vis_pcd2 = o3d.geometry.PointCloud()
    vis_pcd2.points = o3d.utility.Vector3dVector(xyz_down2)

    get_colored_point_cloud_feature_pair(vis_pcd1, 
                                         vis_pcd2,
                                         feature1.detach().cpu().numpy(),
                                         feature2.detach().cpu().numpy(),
                                         config.voxel_size)

    o3d.io.write_point_cloud("../test642.pcd", vis_pcd1)
    o3d.io.write_point_cloud("../test268.pcd", vis_pcd2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--input',
      default='./traindata/oxford',
      type=str,
      help='path to a pointcloud file')
  parser.add_argument(
      '-m',
      '--model',
      default='./outputs/checkpoint.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.2,
      type=float,
      help='voxel size to preprocess point cloud')

  config = parser.parse_args()
  demo(config)