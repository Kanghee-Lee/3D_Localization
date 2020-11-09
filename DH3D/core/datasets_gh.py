# Copyright (C) 2020 Juan Du (Technical University of Munich)
# For more information see <https://vision.in.tum.de/research/vslam/dh3d>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import glob
import random
from sklearn.neighbors import KDTree
import numpy as np

from tensorpack import DataFlow, RNGDataFlow
from tensorpack.dataflow import TestDataSpeed, BatchData, PrefetchDataZMQ
from tensorpack.utils import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

from utils import *
from augment import *


# local train
def get_train_local_selfpair(cfg):
    augmentation = ['Jitter']
    df = Local_train_dataset_selfpair(basedir=cfg.data_basedir, aug=augmentation,
                                      sample_nodes=cfg.sampled_kpnum,
                                      train_file=os.path.join(cfg.data_basedir, 'oxford_train_local_gt.pickle'))
    df = BatchData(df, cfg.batch_size)
    return df


# Global train
def get_train_global_triplet(cfg={}):
    augmentation = ['Jitter', 'RotateSmall', 'Rotate1D']
    if 'data_aug' in cfg:
        augmentation = cfg.data_aug
    df = Global_train_dataset_triplet(basedir=cfg.data_basedir,
                                      train_file=os.path.join(cfg.data_basedir, 'oxford_train_global_gt.pickle'),
                                      posnum=cfg.num_pos, negnum=cfg.num_neg, other_neg=cfg.other_neg,
                                      aug=augmentation)
    df = BatchData(df, cfg.batch_size)
    return df


class Local_test_dataset(DataFlow):
    def __init__(self, basedir, numpts=2 * 8192, knn_require=8, dim=6):
        assert os.path.isdir(basedir)
        self.basedir = basedir
        self.testfile_list = self.get_list()
        self.testfile_num = len(self.testfile_list)
        self.knn = knn_require
        self.numpts = numpts
        self.dim = dim

    def get_list(self):
        pcl_files = glob.glob("{}/*.bin".format(self.basedir))
        pcl_list = sorted(pcl_files)
        print("{} ppointclouds to predict".format(len(pcl_list)))
        return pcl_list

    def __len__(self):
        return len(self.testfile_list)

    def load_test_pc(self, index):
        pcfile = self.testfile_list[index]
        cloud = load_single_pcfile(pcfile, dim=self.dim)
        ori_num = cloud.shape[0]
        if ori_num != self.numpts:
            # downsample is not required if the pointcloud is already processed by the voxelsize around 0.2
            cloud, ori_num = get_fixednum_pcd(cloud, self.numpts, randsample=False, need_downsample=True)
        else:
            choice_idx = np.random.choice(cloud.shape[0], self.numpts, replace=False)
            cloud = cloud[choice_idx, :]

        name = os.path.basename(pcfile)
        ret = [cloud, name, ori_num]
        if self.knn > 0:
            knn_ind, distances = get_knn(cloud, self.knn)
            ret.append(knn_ind)
        return ret

    def __iter__(self):
        for i in range(0, self.testfile_num):
            ret = self.load_test_pc(i)
            yield ret


class Local_train_dataset_selfpair(RNGDataFlow):
    def __init__(self, basedir, train_file, numpts=8192, sample_nodes=256, dim=3, rot_maxv=np.pi, aug=['Jitter'], shuffle=True):
        assert os.path.isdir(basedir)
        self.basedir = basedir
        self.shuffle = shuffle
        self.numpts = numpts
        self.dim = dim
        self.sample_nodes = sample_nodes
        self.rot_maxv = rot_maxv
        self.augmentation = get_augmentations_from_list(aug, upright_axis=2)
        self.dict = get_sets_dict(train_file)
        self.size = len(self.dict.keys())

    def __len__(self):
        return len(self.dict.keys())

    def process_point_cloud(self, cloud):
        cloud, _ = get_fixednum_pcd(cloud, self.numpts, randsample=True, need_downsample=False, sortby_dis=False)
        # augmentation
        for a in self.augmentation:
            cloud = a.apply(cloud)
        return cloud

    def loadPair(self, ind):
        pcfile = self.dict[ind]['query']
        pcfile = os.path.join(self.basedir, pcfile + '.bin')
        cloud = load_single_pcfile(pcfile, dim=self.dim)
        pc1 = self.process_point_cloud(cloud[:, 0:3])  # augmentation
        pc2 = self.process_point_cloud(cloud[:, 0:3])  # augmentation

        ## 1D rotate
        rotation_angle = np.random.uniform(low=-self.rot_maxv, high=self.rot_maxv)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        pc2_trans = np.dot(pc2, rotation_matrix)

        # sample
        farthest_sampler = FarthestSampler()
        pcd1_subset_ind = np.random.choice(pc1.shape[0], pc1.shape[0] // 2, replace=False)
        pcd1_subset = pc1[list(pcd1_subset_ind), :]
        anc_subset_node_inds = farthest_sampler.sample(pcd1_subset, self.sample_nodes)
        anc_node_inds = pcd1_subset_ind[anc_subset_node_inds]
        tree = KDTree(pc2)
        _, pos_node_inds = tree.query(pc1[anc_node_inds, :], k=1)
        pos_node_inds = pos_node_inds.flatten()
        return pc1, pc2_trans, rotation_matrix, anc_node_inds, pos_node_inds

    def __iter__(self):
        fileidxs = list(range(self.size))
        if self.shuffle:
            self.rng.shuffle(fileidxs)
        self.queries_idxs = fileidxs
        for i in self.queries_idxs:
            res = self.loadPair(ind=i)
            yield res


class Global_train_dataset_triplet(RNGDataFlow):
    def __init__(self, basedir, train_file, posnum, negnum, numpts=8192, dim=3,
                 aug=['Jitter', 'RotateSmall', 'Shift', 'Rotate1D'], shuffle=True, randsample=True,
                 other_neg=False):
        assert os.path.isdir(basedir)
        self.basedir = basedir
        self.shuffle = shuffle
        self.numpts = numpts
        self.dim = dim
        self.randsample = randsample
        self.pos_num = posnum
        self.neg_num = negnum
        self.other_neg = other_neg
        self.augmentation = get_augmentations_from_list(aug, upright_axis=2)
        self.dict = get_sets_dict(train_file)
        # print(self.dict.keys())
        # print(self.dict[74])
        # print(self.dict[75])
        # print(self.dict[76])

        self.size = len(self.dict.keys())

    def __len__(self):
        return len(self.dict.keys())

    def load_single_pcfile(filename, dim=3, dtype=np.float32):
        pc = np.fromfile(filename, dtype=dtype)
        pc = np.reshape(pc, (pc.shape[0] // dim, dim))
        return pc[:, 0:3]

    def loadPC(self, ind):
        pcfile = self.dict[ind]['query']
        # fcgf_pcfile = os.path.join(self.basedir, pcfile + '.bin')
        fcgf_pcfile = os.path.join(self.basedir, pcfile + '.npz')
        fcgf = np.load(fcgf_pcfile)
        print('-'*10, 'Load PC', '-'*10)
        print('loadPC points shape : ', fcgf['points'].shape)
        print('loadPC xyz shape : ', fcgf['xyz'].shape)
        print('loadPC feature shape : ', fcgf['feature'].shape)
        print('-' * 30)
        length = fcgf['feature'].shape[0]
        # cloud = load_single_pcfile(pcfile, dim=self.dim)
        # cloud, _ = get_fixednum_pcd(cloud, self.numpts, randsample=True, need_downsample=False, sortby_dis=True)
        # for a in self.augmentation:
        #     cloud = a.apply(cloud)
        return fcgf['points'], fcgf['feature'], length

    def loadPC_list(self, pcdlist):
        pcs = []
        fcgfs = []
        lengths = []
        # for ind in pcdlist:
        #     pc = self.loadPC(ind)
        #     pcs.append(pc)
        for ind in pcdlist:
            fcgf = self.loadPC(ind)
            pcs.append(fcgf[0])
            fcgfs.append(fcgf[1])
            lengths.append(fcgf[2])
        # return listlen * numpts, 3
        pcs = np.concatenate(pcs)
        fcgfs = np.concatenate(fcgfs)
        # print('loadPC_list shape : ', fcgfs.shape)

        return pcs, fcgfs, lengths

    def __iter__(self):
        fileidxs = list(range(self.size))
        if self.shuffle:
            self.rng.shuffle(fileidxs)
        self.queries_idxs = fileidxs
        # print(len(fileidxs)) # 21026
        # print(self.dict[1])
        # print(self.dict[2])
        # print(self.dict[3])

        for i in self.queries_idxs:
            positives = self.dict[i]['positives']
            nonnegtives = self.dict[i]['nonnegtives']
            if len(positives) < self.pos_num:
                continue
            posind = [positives[i] for i in np.random.choice(len(positives), size=self.pos_num, replace=False)]
            possible_negs = list(set(self.dict.keys()) - set(nonnegtives))
            negind = [possible_negs[i] for i in np.random.choice(len(possible_negs), size=self.neg_num, replace=False)]
            # print('posind : ', posind)
            # print('possible_negs : ', possible_negs)
            # print('negind : ', negind)
            query_pcd = self.loadPC(i)
            pos_pcds = self.loadPC_list(posind)
            neg_pcds = self.loadPC_list(negind)
            if not self.other_neg:
                yield [query_pcd, pos_pcds, neg_pcds]
            else:
                # get neighbors of negatives and query
                neighbors = []
                for pos in positives:
                    neighbors.append(pos)
                for neg in negind:
                    for pos in self.dict[neg]["positives"]:
                        neighbors.append(pos)
                # print('neighbors : ', neighbors)
                # print('len(neighbors) : ', len(neighbors))

                possible_negs = list(set(self.dict.keys()) - set(neighbors))
                # print('possible_negs : ', possible_negs)
                random.shuffle(possible_negs)
                otherneg = self.loadPC(possible_negs[0])
                query_pts, query_feat, query_len = query_pcd[0], query_pcd[1], np.array([query_pcd[2]])
                pos_pts, pos_feat, pos_len = pos_pcds[0], pos_pcds[1], np.array(pos_pcds[2])
                neg_pts, neg_feat, neg_len = neg_pcds[0], neg_pcds[1], np.array(neg_pcds[2])
                other_pts, other_feat, other_len = otherneg[0], otherneg[1], np.array([otherneg[2]])
                print(query_pts.shape, query_feat.shape, query_len.shape)   # (6193, 3), (5912, 32), (1, )
                print(pos_pts.shape, pos_feat.shape, pos_len.shape) # (12278, 3), (10846, 32), (2, )
                print(neg_pts.shape, neg_feat.shape, neg_len.shape)
                print(other_pts.shape, other_feat.shape, other_len.shape)

                # print('otherneg : ', otherneg)
                # print('datasets - query_pcd : ', query_pcd['feature'].shape)   # (21801, )
                # print('datasets - pos_pcd : ', pos_pcds['feature'].shape)   # (44082, ) --> 2 pos pcds
                # print('datasets - neg_pcd : ', neg_pcds['feature'].shape)   # (177108, ) --> 8 neg pcds
                # print('datasets - otherneg : ', otherneg['feature'].shape)   # (8922, )

                # yield [query_pcd, pos_pcds, neg_pcds, otherneg]
                yield [query_pts, query_feat, query_len, pos_pts, pos_feat, pos_len, neg_pts, neg_feat, neg_len,
                       other_pts, other_feat, other_len]


class Global_test_dataset(RNGDataFlow):
    def __init__(self, basedir, test_file,
                 numpts=4096 * 2, pcd_dim=3, pcd_dtype=np.float32, **kwarg):
        assert os.path.isdir(basedir)
        assert os.path.isfile(test_file)
        self.basedir = basedir
        self.testdict = get_sets_dict(test_file)
        self.numpts = numpts
        self.pcd_dim = pcd_dim
        self.pcd_dtype = pcd_dtype

        if 'eval_sequences' in kwarg:
            self.eval_sequences = sorted(kwarg.get('eval_sequences'))
        else:
            self.eval_sequences = sorted(self.testdict.keys())
        self.eval_list = self.get_pcd_list()
        self.size = len(self.eval_list)

    def __len__(self):
        return self.size

    def get_pcd_list(self):
        pcdlist = []
        for seq in self.eval_sequences:
            seqinfo = self.testdict[seq]
            for pcd in seqinfo:
                pcdlist.append(pcd['query'] + '.bin')
        print("{} pointclouds to predict. \n".format(len(pcdlist)))
        return pcdlist

    def __iter__(self):
        for i in range(self.size):
            name = self.eval_list[i]
            pcd = load_single_pcfile(os.path.join(self.basedir, name), dim=self.pcd_dim, dtype=self.pcd_dtype)

            if pcd.shape[0] != self.numpts:
                pcd, ori_num = get_fixednum_pcd(pcd, self.numpts, randsample=True, need_downsample=False,
                                                sortby_dis=True)
            yield [pcd, name]


if __name__ == '__main__':
    pass
