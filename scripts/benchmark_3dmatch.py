"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d
import pickle
from tabulate import tabulate
import math
from scipy import spatial
from collections import namedtuple
from scipy.spatial import cKDTree
from lib.timer import Timer, AverageMeter

from util.misc import extract_features, extract_globalDesc, extract_globalDesc_from_fcgf

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, \
  gather_results, global_desc_matching, read_dataAsnumpy
import torch

import MinkowskiEngine as ME
from config import get_config
from easydict import EasyDict as edict

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

###################################################################################################
# class GlobalDesc_eval(object):
#
#   def __init__(self, result_savedir, database_file, query_file, desc_dir, max_num_nn=25, dim=256, *arg, **kwarg):
#     super(GlobalDesc_eval, self).__init__()
#     assert (os.path.isfile(database_file) and os.path.isfile(query_file))
#     self.desc_dir = desc_dir
#     self.database_sets = get_sets_dict(database_file)
#     self.query_sets = get_sets_dict(query_file)
#     self.database_seqnum = len(self.database_sets)
#     self.query_seqnum = len(self.query_sets)
#     self.max_querynum = max_num_nn
#     self.savedir = result_savedir
#     self.desc_dim = dim
#     if 'database_sequences' in kwarg:
#       self.database_sequences = sorted(kwarg.get('database_sequences'))
#     else:
#       self.database_sequences = sorted(self.database_sets.keys())
#     if 'query_sequences' in kwarg:
#       self.query_sequences = sorted(kwarg.get('query_sequences'))
#     else:
#       self.query_sequences = sorted(self.query_sets.keys())
#
#     print("loading databse position info and descriptors")
#     self.database_pos, self.database_desc = self.get_database_pos_desc(isquery=False)
#     print(" {} databases loaded".format(len(self.database_pos)))
#
#     print("loading query position info and descriptors")
#     self.query_pos, self.query_desc = self.get_database_pos_desc(True)
#     print("{} queries loaded".format(len(self.query_pos)))
#
#     print(tabulate(
#       [[self.database_sequences[i], len(self.database_desc[i])] for i in range(len(self.database_sequences))],
#       headers=["databaseseq", "segnum"]))
#     print()
#     print(tabulate([[self.query_sequences[i], len(self.query_desc[i])] for i in range(len(self.query_sequences))],
#                    headers=["queryseq", "segnum"]))
#
#   def get_database_pos_desc(self, isquery):
#     pos_sets = []
#     desc_sets = []
#     descdir = self.desc_dir
#     ext = '.bin'
#     if isquery:
#       usedict = self.query_sets
#       useseq = self.query_sequences
#
#     else:
#       usedict = self.database_sets
#       useseq = self.database_sequences
#
#     for seq in useseq:
#       seqinfo = usedict[seq]
#       # print("{} has {} pointclouds \n".format(seq, len(seqinfo)))
#       pos = {'northing': [], 'easting': []}
#       descriptors = []
#       for pcd in seqinfo:
#         pos['northing'].append(pcd['northing'])
#         pos['easting'].append(pcd['easting'])
#         pcd_filepath = pcd['query']
#         desc = load_descriptor_bin(os.path.join(descdir, pcd_filepath + ext), self.desc_dim)
#         descriptors.append(desc)
#       pos_sets.append(pos)
#       descriptors = np.vstack(descriptors)
#       desc_sets.append(descriptors)
#     print(len(desc_sets))
#     print(desc_sets[0].shape)
#     return pos_sets, desc_sets
#

#
#
# if __name__ == '__main__':
#   ref_file = "../data/oxford_test_global/oxford_test_global_gt_reference.pickle"
#   query_file = "../data/oxford_test_global/oxford_test_global_gt_query.pickle"
#   res_dir = "../data/global/globaldesc_results"
#
#   retrieval = GlobalDesc_eval(result_savedir='./', desc_dir=res_dir,
#                               database_file=ref_file, query_file=query_file,
#                               database_sequences=['2015-03-10-14-18-10'],
#                               query_sequences=['2015-11-13-10-28-08'])
#   retrieval.evaluate()
###################################################################################################
def get_sets_dict(filename):
  with open(filename, 'rb') as handle:
    file_dict = pickle.load(handle)
    # print("number of item: {}.\n".format(len(trajectories.keys())))
    return file_dict


def get_database_desc(desc_dir, database_sequences, database_sets, isquery):
  desc_sets = []
  if isquery :
    for seq in database_sequences:
      seqinfo = database_sets[seq]
      # print("{} has {} pointclouds \n".format(seq, len(seqinfo)))

      descriptors = []
      for i in seqinfo:

        ext = "%s_%03d" % (seq, i)

        desc = read_dataAsnumpy(desc_dir, ext)

        descriptors.append(desc)
      descriptors = np.vstack(descriptors)
      desc_sets.append(descriptors)
    print('get_database_desc - desc_sets length : ', len(desc_sets))
    for i in range(len(desc_sets)):
      print('get_database_desc - i th desc_sets shape : ', desc_sets[i].shape)

    return desc_sets
  else :
    for seq in database_sequences:
      seqinfo = database_sets[seq]
      # print("{} has {} pointclouds \n".format(seq, len(seqinfo)))
      print('get_database_desc - seqinfo length : ', len(seqinfo))


      descriptors = []
      for i in range(len(seqinfo)):

        ext = "%s_%03d" % (seq, i)

        desc=read_dataAsnumpy(desc_dir, ext)

        descriptors.append(desc)
      descriptors = np.vstack(descriptors)
      desc_sets.append(descriptors)
    print('get_database_desc - desc_sets length : ', len(desc_sets))
    for i in range(len(desc_sets)):
      print('get_database_desc - i th desc_sets shape : ', desc_sets[i].shape)

    return desc_sets


def retrieval(ref_descriptors, query_descriptors, max_num_nn):
  ref_tree = cKDTree(ref_descriptors)
  _, indices = ref_tree.query(query_descriptors, k=max_num_nn)
  #
  # print(_)
  # print(indices)
  # print(ref_descriptors[0])
  # print(query_descriptors[1])
  # print(np.sqrt(np.power(ref_descriptors[0]-query_descriptors[1], 2).sum(0)))
  # assert 0
  return indices

def compute_tp_fp(ref_descriptors, query_descriptors,
                  gt_matches, query_inds,*arg, **kwarg):
  threshold = max(int(round(len(ref_descriptors) / 10.0)), 1)
  temp_indices = retrieval(ref_descriptors, query_descriptors, *arg, **kwarg)
  indices=[]
  for i in range(len(query_inds)) :
    a = np.delete(temp_indices[i], np.argwhere(temp_indices[i]==query_inds[i]))
    indices.append(a)
  indices=np.vstack(indices)
  tp = gt_matches[np.expand_dims(np.arange(len(indices)), axis=1), indices]
  fp = np.logical_not(tp)
  tp_cum = np.cumsum(tp, axis=1)
  fp_cum = np.cumsum(fp, axis=1)
  valid = np.any(gt_matches, axis=1)
  one_percent_retrieved = np.any(tp[:, 0:threshold], axis=1)
  return tp_cum, fp_cum, valid, one_percent_retrieved


def evaluate(desc_dir, source):
  # save_fn = "%s_%03d" % (fo_base, i)
  retrieval_Result = namedtuple(
    'RetrievResult', ['refseq', 'queryseq', 'recalls', 'one_percent_retrieved'])

  results = []

  database_file = os.path.join(source, 'gt_matches.pickle')
  query_file = os.path.join(source, 'gt_query.pickle')

  database_sets = get_sets_dict(database_file)
  query_sets = get_sets_dict(query_file)

  database_sequences = list(database_sets.keys())
  query_sequences = list(query_sets.keys())
  # print(database_sets)
  # print('-'*20)
  # print(query_sets)
  database_desc=get_database_desc(desc_dir, database_sequences, database_sets, False)
  print('-'*50)
  query_desc = get_database_desc(desc_dir, query_sequences, query_sets, True)
############################################################################################################
  # 하이퍼 파라미터 조정을 위해 positive, negative threshold 분석 #
  # pos_dist=[]
  # neg_dist=[]
  # for ref_ind in range(len(database_sequences)):
  #   print(database_sequences[ref_ind])
  #
  #   pcd_key = database_sequences[ref_ind]
  #   print('evaluate - query_sets[pcd_key] : ', query_sets[pcd_key])
  #   query_ind = ref_ind
  #   gt_matches = database_sets[pcd_key][query_sets[pcd_key]]
  #   qu_desc = query_desc[query_ind]
  #   print(len(database_sets[pcd_key]))
  #   db_desc = database_desc[ref_ind]
  #   l = len(database_sets[pcd_key])
  #   for i in range(l) :
  #     for j in range(i+1, l) :
  #       a = db_desc[i]
  #       b = db_desc[j]
  #       temp_dist = np.sqrt(np.power((a - b), 2).sum(0))
  #
  #       # print(i, j, gt_matches[i][j])
  #       if gt_matches[i][j] ==0 :
  #         neg_dist.append([i, j, temp_dist])
  #       else :
  #         pos_dist.append([i, j, temp_dist])
  #
  # print(len(pos_dist))
  # print(len(neg_dist))
  # pos_dist = np.vstack(pos_dist)
  # neg_dist = np.vstack(neg_dist)
  # print(pos_dist[:, -1].mean(0))
  # print(neg_dist[:, -1].mean(0))
  # assert 0

############################################################################################################
  for ref_ind in range(len(database_sequences)):
    db_desc = database_desc[ref_ind]
    for query_ind in range(len(query_sequences)):
      if database_sequences[ref_ind] != query_sequences[query_ind]:
        continue

      pcd_key = database_sequences[ref_ind]
      print('evaluate - query_sets[pcd_key] : ', query_sets[pcd_key])

      qu_desc = query_desc[query_ind]
      gt_matches = database_sets[pcd_key][query_sets[pcd_key]]
      tp, fp, valid, one_percent = compute_tp_fp(db_desc, qu_desc, gt_matches,query_sets[pcd_key],
                                                 max_num_nn=26)
      recall = np.mean(tp[valid] > 0, axis=0)
      validoneperc = np.mean(one_percent[valid])
      print(recall)
      ret = retrieval_Result(database_sequences[ref_ind], query_sequences[query_ind], recall,
                             validoneperc)
      results.append(ret)
  print(tabulate(
    [[ret.refseq, ret.queryseq, ret.recalls[0:5], ret.recalls[5:10], ret.recalls[10:15], ret.one_percent_retrieved] for
     ret in
     results], floatfmt=".4f",
    headers=["refseq", "queryseq", "recalls0-5", "recalls5-10", "recalls10-15", "1%"]))

  recalls = np.vstack([ret.recalls for ret in results])
  one_percent_retrieved = np.hstack([ret.one_percent_retrieved for ret in results])

  avg_recall = np.mean(recalls, axis=0)
  avg_one_percent_retrieved = np.mean(one_percent_retrieved)

  print("\n")
  print("Avg_recall:")
  for i, r in enumerate(avg_recall):
    print("{}: {:.4f}".format(i + 1, r))
  print("\n")
  print("Avg_one_percent_retrieved:")
  print("{:.4f}".format(avg_one_percent_retrieved))

  assert 0
  for pcd_key in query_sequences :
    gt_matches = database_sets[pcd_key][query_sets[pcd_key]]


  for ref_ind in range(len(database_sequences)):
    print(database_sequences)
    print(database_sequences[ref_ind])
    print(query_sequences)
    if database_sequences[ref_ind] not in query_sequences :
      continue
    pcd_key=database_sequences[ref_ind]

    print(query_sets[pcd_key])
    print(database_sets[pcd_key])
    gt_matches = database_sets[pcd_key][query_sets[pcd_key]]

    assert 0
  #
  # for query_ind in range(len(query_sequences)):
  #
  # gt_matches =
  # assert 0




  retrieval_Result = namedtuple(
    'RetrievResult', ['refseq', 'queryseq', 'recalls', 'one_percent_retrieved'])

  results = []
  for ref_ind in range(len(database_sequences)):
    ref_desc = database_desc[ref_ind]

    for query_ind in range(len(self.query_sequences)):
      if self.database_sequences[ref_ind] == self.query_sequences[query_ind]:
        continue
      query_loc = self.query_pos[query_ind]
      query_desc = self.query_desc[query_ind]
      gt_matches = is_gt_match_2D(query_loc, ref_loc, 25)

      tp, fp, valid, one_percent = compute_tp_fp(ref_desc, query_desc, gt_matches,
                                                 max_num_nn=self.max_querynum)
      recall = np.mean(tp[valid] > 0, axis=0)
      validoneperc = np.mean(one_percent[valid])
      ret = retrieval_Result(self.database_sequences[ref_ind], self.query_sequences[query_ind], recall,
                             validoneperc)
      results.append(ret)
  print(tabulate(
    [[ret.refseq, ret.queryseq, ret.recalls[0:5], ret.recalls[5:10], ret.recalls[10:15], ret.one_percent_retrieved]
     for ret in
     results], floatfmt=".4f",
    headers=["refseq", "queryseq", "recalls0-5", "recalls5-10", "recalls10-15", "1%"]))

  recalls = np.vstack([ret.recalls for ret in results])
  one_percent_retrieved = np.hstack([ret.one_percent_retrieved for ret in results])

  avg_recall = np.mean(recalls, axis=0)
  avg_one_percent_retrieved = np.mean(one_percent_retrieved)

  print("\n")
  print("Avg_recall:")
  for i, r in enumerate(avg_recall):
    print("{}: {:.4f}".format(i + 1, r))
  print("\n")
  print("Avg_one_percent_retrieved:")
  print("{:.4f}".format(avg_one_percent_retrieved))

def make_gt_pickle(source_path, target_path, device):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  #-
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  #-
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0

  gt_all_dict = {}
  gt_pos_dict={}
  gt_matches={}
  gt_query = {}
  for fo in folders:
    # gt.log file is in 'evaluation' folder
    if 'evaluation' not in fo:
      continue
    files = os.path.join(fo, 'gt.log')
    print('-'*20)
    pcd_key = fo.rsplit('-evaluation', 1)[0].rsplit('/', 1)[1]
    print(pcd_key)
    # pcd_default = '/cloud_bin_'
    # pcd_format = '.ply'
    with open(files) as f:
      content = f.readlines()
      lines = [x.strip().split() for x in content]
      length = int(lines[0][2])
      pos_pairs = [[int(lines[i][0]), int(lines[i][1])] for i in range(len(lines)) if i%5==0]
      # print(pair_lines)

      all_pairs=[]
      for i in range(length) :
        for j in range(i+1, length) :
          if [i, j] in pos_pairs :
            all_pairs.append([i, j, 1])
          else :
            all_pairs.append([i, j, 0])
    matches = np.zeros((length, length))
    queries = np.array([i for i in range(length)])
    queries2 = np.zeros((length))
    for p in pos_pairs :
      i, j = p
      queries2[i] +=1
      queries2[j] += 1
    # print(queries2)
    s=queries2.argsort()
    s=s[::-1]
    # print(pos_pairs)
    # print(s)

    for p in pos_pairs :
      i, j = p
      matches[i, j] = 1


    gt_matches[pcd_key] = matches
    gt_all_dict[pcd_key] = all_pairs
    gt_pos_dict[pcd_key] = pos_pairs
    gt_query[pcd_key] = queries

    # print(length)
    # print(len(all_pairs))
    # print(len(pos_pairs))
    # print(gt_all_dict)

    #-- Make overlap.txt file for training with test data--#
  #   save_txt = "%s-0.30.txt" % (pcd_key)
  #   overlap = open(os.path.join(source_path, save_txt), 'w')
  #   for i, j in pos_pairs :
  #     save_fn1 = "%s_%03d.npz" % (pcd_key, i)
  #     save_fn2 = "%s_%03d.npz" % (pcd_key, j)
  #     save_fn = save_fn1 + " " + save_fn2
  #     overlap.write("%s\n" % (save_fn))
  # assert 0

  with open(os.path.join(source_path, 'gt_matches.pickle'), 'wb') as f:
    pickle.dump(gt_matches, f)
  with open(os.path.join(source_path, 'gt_query.pickle'), 'wb') as f:
    pickle.dump(gt_query, f)
  #
  # print(gt_matches)
  #
  # database_sets = gt_matches
  #
  # database_sequences = list(database_sets.keys())
  # print(database_sets)
  # print(database_sequences)
  #
  # for seq in database_sequences:
  #   seqinfo = database_sets[seq]
  #   # print("{} has {} pointclouds \n".format(seq, len(seqinfo)))
  #   print('get_database_desc - seqinfo length : ', len(seqinfo))
  #
  #   descriptors = []
  #   for i in range(len(seqinfo)):
  #     ext = "%s_%03d" % (seq, i)
  #
  #     desc = read_dataAsnumpy(desc_dir, ext)
  #
  #     descriptors.append(desc)
  #   descriptors = np.vstack(descriptors)
  #   desc_sets.append(descriptors)
  # print('get_database_desc - desc_sets length : ', len(desc_sets))
  # for i in range(len(desc_sets)):
  #   print('get_database_desc - i th desc_sets shape : ', desc_sets[i].shape)
  #
  # return desc_sets
  # assert 0

###################################################################################################

  #
  #   pos_fnames = [x.strip().split() for x in content]
  #   for pos_fname in pos_fnames:
  #     self.pos_files.append([pos_fname[0], pos_fname[1]])
  #   print(files)
  #   assert 0
  #   fo_base = os.path.basename(fo)
  #   f.write("%s %d\n" % (fo_base, len(files)))
  #   for i, fi in enumerate(files):
  #     # Extract features from a file
  #     pcd = o3d.io.read_point_cloud(fi)
  #     save_fn = "%s_%03d" % (fo_base, i)
  #     if i % 100 == 0:
  #       logging.info(f"{i} / {len(files)}: {save_fn}")
  #
  #     timer.tic()
  #     xyz_down, feature = extract_globalDesc(
  #         model,
  #         xyz=np.array(pcd.points),
  #         rgb=None,
  #         normal=None,
  #         voxel_size=voxel_size,
  #         device=device,
  #         skip_check=True)
  #     t = timer.toc()
  #     if i > 0:
  #       tmeter.update(t)
  #       num_feat += len(xyz_down)
  #
  #     np.savez_compressed(
  #         os.path.join(target_path, save_fn),
  #         points=np.array(pcd.points),
  #         xyz=xyz_down,
  #         feature=feature.detach().cpu().numpy())
  #     if i % 20 == 0 and i > 0:
  #       logging.info(
  #           f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
  #       )
  #
  # f.close()

###################################################################################################

def extract_globalDesc_batch(model, config, source_path, target_path, voxel_size, device):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      pcd = o3d.io.read_point_cloud(fi)
      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")

      timer.tic()
      xyz_down, feature = extract_globalDesc(
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

  f.close()


###################################################################################################

###################################################################################################

def extract_globalDesc_batch_from_fcgf(model, config, source_path, feature_path, target_path, voxel_size, device):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file

      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")
      fcgf = read_dataAsnumpy(feature_path, save_fn)
      fcgf_length = [fcgf.shape[0]]
      timer.tic()
      # feature = extract_globalDesc_from_fcgf(
      #   model,
      #   fcgf,
      #   fcgf_length,
      #   device=device)
      # GG
      feature,_,_ = extract_globalDesc_from_fcgf(
          model,
          fcgf,
          fcgf_length,
          device=device)
      t = timer.toc()
      if i > 0:
        tmeter.update(t)

      np.savez_compressed(
          os.path.join(target_path, save_fn),
          feature=feature.detach().cpu().numpy())
      if i % 20 == 0 and i > 0:
        logging.info(
            f'Average time: {tmeter.avg}, FPS: {tmeter.sum}, time / feat: {tmeter.sum}, '
        )

  f.close()


###################################################################################################





def extract_features_batch(model, config, source_path, target_path, voxel_size, device):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      pcd = o3d.io.read_point_cloud(fi)
      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")

      timer.tic()
      xyz_down, feature, length = extract_features(
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
          feature=feature.detach().cpu().numpy(),
          length=length)
      if i % 20 == 0 and i > 0:
        logging.info(
            f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
        )

  f.close()


def registration(feature_path, voxel_size, pooling_arg):
  """
  Gather .log files produced in --target folder and run this Matlab script
  https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
  (see Geometric Registration Benchmark section in
  http://3dmatch.cs.princeton.edu/)
  """
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  for s in sets:
    set_name = s[0]
    pts_num = int(s[1])
    matching_pairs = gen_matching_pair(pts_num)
    results = []
    for m in matching_pairs:
      if pooling_arg =='None' :
        results.append(do_single_pair_matching(feature_path, set_name, m, voxel_size))
      else :
        results.append(global_desc_matching(feature_path, set_name, m, voxel_size, pooling_arg))
      print("results :", results)
    traj = gather_results(results)
    logging.info(f"Writing the trajectory to {feature_path}/{set_name}.log")
    write_trajectory(traj, "%s.log" % (os.path.join(os.path.join(feature_path, 'global'), set_name)))


def do_single_pair_evaluation(feature_path,
                              set_name,
                              traj,
                              voxel_size,
                              tau_1=0.1,
                              tau_2=0.05,
                              num_rand_keypoints=-1):
  trans_gth = np.linalg.inv(traj.pose)
  i = traj.metadata[0]
  j = traj.metadata[1]
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)

  # coord and feat form a sparse tensor.
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

  # use the keypoints in 3DMatch
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

  # logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
  if hit_ratio >= tau_2:
    return True
  else:
    return False


def feature_evaluation(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]

  assert len(
      sets
  ) > 0, "Empty list file. Makesure to run the feature extraction first with --do_extract_feature."

  tau_1 = 0.1  # 10cm
  tau_2 = 0.05  # 5% inlier
  logging.info("%f %f" % (tau_1, tau_2))
  recall = []
  for s in sets:
    set_name = s[0]
    print('*'*20)
    print(source_path)
    print(set_name)
    print('*'*20)
    traj = read_trajectory(os.path.join(source_path, set_name + "-evaluation/gt.log"))
    assert len(traj) > 0, "Empty trajectory file"
    results = []
    for i in range(len(traj)):
      results.append(
          do_single_pair_evaluation(feature_path, set_name, traj[i], voxel_size, tau_1,
                                    tau_2, num_rand_keypoints))

    mean_recall = np.array(results).mean()
    std_recall = np.array(results).std()
    recall.append([set_name, mean_recall, std_recall])
    logging.info(f'{set_name}: {mean_recall} +- {std_recall}')
  for r in recall:
    logging.info("%s : %.4f" % (r[0], r[1]))
  scene_r = np.array([r[1] for r in recall])
  logging.info("average : %.4f +- %.4f" % (scene_r.mean(), scene_r.std()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to 3dmatch test dataset')
  parser.add_argument(
      '--source_high_res',
      default=None,
      type=str,
      help='path to high_resolution point cloud')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
    '--feature', default=None, type=str, help='path to saved FCGF')
  parser.add_argument(
      '-m',
      '--model_path',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', action='store_true')
  parser.add_argument('--extract_globalDesc', action='store_true')
  parser.add_argument('--extract_globalDesc_from_fcgf', action='store_true')
  parser.add_argument('--evaluate_feature_match_recall', action='store_true')
  parser.add_argument(
      '--evaluate_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument('--with_cuda', action='store_true')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=5000,
      help='Number of random keypoints for each scene')
  parser.add_argument('--max_pooling', action='store_true')
  parser.add_argument('--avg_pooling', action='store_true')
  parser.add_argument('--model', default='ResUnetMLP2', type=str, help='model name')
  parser.add_argument('--make_gt_pickle', action='store_true')
  parser.add_argument('--evaluate', action='store_true')

  parser.add_argument(
    '--desc_dir', default=None, type=str, help='path to load saved global descriptors')
  # parser.add_argument(
  #   '--database_file', default=None, type=str, help='path to load gt database')
  # parser.add_argument(
  #   '--query_file', default=None, type=str, help='path to load query database')

  args = parser.parse_args()

  device = torch.device('cuda' if args.with_cuda else 'cpu')
  #-
  if args.make_gt_pickle:
    make_gt_pickle(args.source, args.target, device)

  if args.evaluate:
    evaluate(args.desc_dir, args.source)
  #-
  if args.extract_features:
    assert args.model_path is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(args.model)
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

####################################################################################

  if args.extract_globalDesc:
    assert args.model_path is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(args.model)
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
      extract_globalDesc_batch(model, config, args.source, args.target, config.voxel_size,
                             device)


  if args.extract_globalDesc_from_fcgf:
    assert args.model_path is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    print('-'*20)
    print(epoch, 'epoch trained')
    print('-' * 20)
    num_feats = 1
    Model = load_model(args.model)
    model = Model()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    with torch.no_grad():
      extract_globalDesc_batch_from_fcgf(model, config, args.source, args.feature, args.target, config.voxel_size,
                             device)

  if args.evaluate_registration:
    assert (args.target is not None)
    with torch.no_grad():
      if args.max_pooling : 
        registration(args.target, args.voxel_size, 'max')
      elif args.avg_pooling : 
        registration(args.target, args.voxel_size, 'avg')
      else :
        registration(args.target, args.voxel_size, 'None')


