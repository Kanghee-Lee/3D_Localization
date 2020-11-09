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
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

import losses
import backbones
from layers import knn_bruteforce
from tf_utils import backbone_scope
from tf_ops.grouping.tf_grouping import group_point

class DH3D(ModelDesc):

    def __init__(self, config):
        super(ModelDesc, self).__init__()
        self.config = config
        self.input_knn_indices = True if self.config.num_points > 8192 else False  # # if num_points > 8192 since the maximum
        # number of knn_bruteforce is 8192

        ## fcgf
        self.isfcgf = self.config.isfcgf

        ## local
        self.local_backbone = self.config.local_backbone
        self.detection_block = self.config.detection_block

        ## global
        self.global_backbone = self.config.global_backbone
        self.global_assemble = self.config.global_assemble

        ## loss
        self.local_loss_func = self.config.local_loss
        self.global_loss_func = self.config.global_loss
        self.detection_loss_func = self.config.detection_loss

    @property
    def training(self):
        return bool(get_current_tower_context().is_training)

    def inputs(self):
        # anc, pos, R, ind1, ind2, knn
        # pointclouds
        ret = [tf.TensorSpec((self.config.batch_size, None, 3), tf.float32, 'anchor_pts')]

        ret.append(
            tf.TensorSpec((self.config.batch_size, None, 32), tf.float32,
                          'anchor_feat'))
        ret.append(
            tf.TensorSpec((self.config.batch_size, 1), tf.int32,
                          'anchor_len'))
        if self.config.num_pos > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 3), tf.float32,
                              'pos_pts'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 32), tf.float32,
                              'pos_feat'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, 2), tf.int32,
                              'pos_len'))
        if self.config.num_neg > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 3), tf.float32,
                              'neg_pts'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 32), tf.float32,
                              'neg_feat'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, 8), tf.int32,
                              'neg_len'))
        if self.config.other_neg:
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 3), tf.float32,
                              'otherneg_pts'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, None, 32), tf.float32,
                              'otherneg_feat'))
            ret.append(
                tf.TensorSpec((self.config.batch_size, 1), tf.int32,
                              'otherneg_len'))

        # rotation for local training
        if self.config.input_R:
            ret.append(tf.TensorSpec((self.config.batch_size, 3, 3), tf.float32, 'R'))

        # random indices from local training
        if self.config.sampled_kpnum > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, self.config.sampled_kpnum), tf.int32, 'sample_ind_anchor'))
            ret.append(tf.TensorSpec((self.config.batch_size, self.config.sampled_kpnum), tf.int32, 'sample_ind_pos'))

        # knn indices from input
        if self.config.num_points > 8192:
            ret.append(tf.TensorSpec((self.config.batch_size, self.config.num_points, self.config.knn_num), tf.int32,
                                     'knn_ind_anchor'))
            if self.config.num_pos > 0:
                ret.append(tf.TensorSpec(
                    (self.config.batch_size, self.config.num_points * self.config.num_pos, self.config.knn_num),
                    tf.int32,
                    'knn_ind_pos'))
            if self.config.num_neg > 0:
                ret.append(tf.TensorSpec(
                    (self.config.batch_size, self.config.num_points * self.config.num_neg, self.config.knn_num),
                    tf.int32,
                    'knn_ind_neg'))
        return ret

    def compute_local(self, points, isfreeze=False):
        with backbone_scope(freeze=isfreeze):
            inputs_dict = {
                'points': points,
                'featdim': self.config.featdim,
                'knn_ind': self.knn_indices,
                'dilate': self.config.dilate
            }
            newpoints, localdesc = getattr(backbones, self.local_backbone)(**inputs_dict)
        return newpoints, localdesc



    def compute_global(self, outs, freeze_global=False):
        with backbone_scope(freeze=freeze_global):
            points = outs['xyz']
            localdesc = outs['feat']
            newpoints, forglobal = getattr(backbones, self.global_backbone)(points, localdesc, **self.config)

            ## if sample
            if self.config.global_subsample > 0:
                newpoints, forglobal, kp_indices = backbones.subsample(newpoints, forglobal, self.global_subsample,
                                                                       kp_idx=None)
            # global attention
            global_att = backbones.globalatt_block_fcgf(forglobal, scope="globalatt", ac_func=BNReLU)

            inputs_dict = {
                'xyz': newpoints,
                'features': forglobal,
                'att': global_att,
                'is_training': self.training,
                'add_batch_norm': self.config.add_batch_norm,
            }
            globaldesc = getattr(backbones, self.global_assemble)(**inputs_dict)
        return globaldesc
############################################################ FCGF + DH3D
    def compute_global_fcgf_dh3d(self, outs, freeze_global=False):
        with backbone_scope(freeze=freeze_global):
            points = outs['xyz']
            localdesc = outs['feat']
            length = outs['length']
            # newpoints, forglobal = getattr(backbones, self.global_backbone)(points, localdesc, **self.config)
            # print('forglobal shape : ', forglobal.shape)  # before : (24, 8192, 256)
            print('forglobal shape : ', localdesc.shape)  # after : (?, 32)
            ## if sample
            if self.config.global_subsample > 0:
                newpoints, forglobal, kp_indices = backbones.subsample(points, localdesc, self.global_subsample,
                                                                       kp_idx=None)
                print('#############subsample!!!!##########')
            # global attention
            global_att = backbones.globalatt_block_fcgf(localdesc, scope="globalatt", ac_func=BNReLU)
            print('global_att shape : ', global_att.shape)  # before : (24, 8192, 1) / after : (1, ?, 1)

            inputs_dict = {
                # 'xyz': newpoints,
                'xyz': points,
                'features': localdesc,
                'length' : length,
                'att': global_att,
                'is_training': self.training,
                'add_batch_norm': self.config.add_batch_norm,
            }
            globaldesc = getattr(backbones, self.global_assemble)(**inputs_dict)
            print('globaldesc shape : ', globaldesc.shape)
        return globaldesc
############################################################ FCGF + DH3D
############################################################ FCGF
    def compute_global_fcgf(self, outs, freeze_global=False):
        with backbone_scope(freeze=freeze_global):
            points = outs['xyz']
            localdesc = outs['feat']
            length = outs['length']

            # global attention
            globaldesc = backbones.att_k(localdesc, length, self.training,  scope="att_k", ac_func=BNReLU)

        return globaldesc

############################################################ FCGF



    def build_graph(self, *inputs_dict):
        inputs_dict = dict(zip(self.input_names, inputs_dict))
        # print(inputs_dict)

        ######################### X points
        ####### concat pointclouds
        # pcdset = [inputs_dict['anchor_pts']]
        # if self.config.num_pos > 0:
        #     pcdset.append(tf.reshape(inputs_dict['pos_pts'], [-1, self.config.num_points, 3]))
        # if self.config.num_neg > 0:
        #     pcdset.append(tf.reshape(inputs_dict['neg_pts'], [-1, self.config.num_points, 3]))
        # if self.config.other_neg:
        #     pcdset.append(inputs_dict['otherneg_pts'])
        # points = tf.concat(pcdset, 0, name='pointclouds')  # query+pos+neg+otherneg, numpts, 3
        # print('model.py - points : ', points)
        # print('points : ', points.shape)   # (24, 8192, 3)
        #
        ######################### X points
        if self.input_knn_indices:
            knn_ind_set = [inputs_dict['knn_ind_anchor']]
            if inputs_dict.get('knn_ind_pos'):
                knn_ind_set.append(inputs_dict['knn_ind_pos'])
            if inputs_dict.get('knn_ind_neg'):
                knn_ind_set.append(inputs_dict['knn_ind_neg'])
            knn_inds = tf.concat(knn_ind_set, 0, name='knn_inds')
            self.knn_indices = tf.transpose(knn_inds, perm=[0, 2, 1])  # batch, k. numpts
        # else:
        #     self.knn_indices, distances = knn_bruteforce(tf.transpose(points, perm=[0, 2, 1]), k=self.config.knn_num)
        else :
            self.knn_indices = []
            print('No points')

        if self.config.sampled_kpnum > 0:
            sample_nodes_concat = tf.concat([inputs_dict['sample_ind_anchor'], inputs_dict['sample_ind_pos']], 0)
            self.sample_nodes_concat = tf.expand_dims(sample_nodes_concat, 2)
        else:
            self.sample_nodes_concat = None

        freeze_local = self.config.freezebackbone
        freeze_det = self.config.freezedetection
        freeze_global = self.config.freezeglobal


        ####### get local features
        outs = {}
        outs['xyz'] = []
        outs['knn_indices'] = self.knn_indices
        # newpoints, localdesc = self.compute_local(points, freeze_local)
        # print('model.py - localdesc', localdesc)   # (24, 8192, 128)
        localdesc = [tf.reshape(inputs_dict['anchor_feat'], [-1, 32])]
        lengths = [tf.reshape(inputs_dict['anchor_len'], [-1, 1])]
        if self.config.num_pos > 0:
            localdesc.append(tf.reshape(inputs_dict['pos_feat'], [-1, 32]))
            lengths.append(tf.reshape(inputs_dict['pos_len'], [-1, 1]))
        if self.config.num_neg > 0:
            localdesc.append(tf.reshape(inputs_dict['neg_feat'], [-1, 32]))
            lengths.append(tf.reshape(inputs_dict['neg_len'], [-1, 1]))
        if self.config.other_neg:
            localdesc.append(tf.reshape(inputs_dict['otherneg_feat'], [-1, 32]))
            lengths.append(tf.reshape(inputs_dict['otherneg_len'], [-1, 1]))
        for i in range(len(localdesc)) :
            print('localdesc : ', localdesc[i].shape)
            print('lengths : ', lengths[i].shape)
        localdescs = tf.concat(localdesc, 0, name='localdescs')  #
        lengths = tf.concat(lengths, 0, name='lengths')  #

        print('model.py - localdescs shape : ', localdescs.shape)
        outs['feat'] = localdescs
        # with tf.Session() as sess :
        #     lengths = sess.run(lengths)
        # print(lengths)
        # assert 0
        outs['length'] = lengths
        # localdesc = inputs_dict[]
        # outs['feat'] = localdesc

        ######################### X local
        #
        # if self.config.input_R:
        #     outs['R'] = inputs_dict['R']
        #
        # newpoints, localdesc = self.compute_local(points, freeze_local)
        # print('model.py - localdesc', localdesc)   # (24, 8192, 128)
        # '''
        # 여기서 local desc shape한번 보고 이에 맞게 localfcgf도 고쳐주고 compute global 그대로 돌리면 될듯.
        # issue 1 : detect attention은??
        # '''
        # # assert 0
        # # localfcgf = points
        # localdesc_l2normed = tf.nn.l2_normalize(localdesc, dim=2, epsilon=1e-8, name='feat_l2normed')
        # outs['feat'] = localdesc
        # outs['local_desc'] = localdesc_l2normed
        # print('out - feat : ', outs['feat'].shape)    # (24, 8192, 128)
        # # assert 0
        # saved_tensor_xyz_feat = tf.concat([newpoints, localdesc_l2normed], -1, name='xyz_feat')
        #
        #
        # ####### get local attentions
        # if self.config.detection:
        #     detect_att = getattr(backbones, self.detection_block)(localdesc, freeze_det=freeze_det)
        #     outs['attention'] = detect_att
        #     saved_tensor_xyz_feat_att = tf.concat([newpoints, localdesc_l2normed, detect_att], -1, name='xyz_feat_att')
        #
        # if self.config.sampled_kpnum > 0:
        #     outs['sample_nodes_concat'] = self.sample_nodes_concat
        #     localxyzsample, localfeatsample, kp_indices = backbones.subsample(points, localdesc_l2normed,
        #                                                                           self.config.sampled_kpnum,
        #                                                                           kp_idx=self.sample_nodes_concat)
        #     outs['feat_sampled'] = localfeatsample
        #     outs['xyz_sampled'] = localxyzsample
        #     xyz_feat = tf.concat([localxyzsample, localfeatsample], -1, name='xyz_feat_sampled')
        #     if self.config.get('detection'):
        #         att_sampled = tf.squeeze(group_point(detect_att, kp_indices), axis=-1)
        #         outs['att_sampled'] = att_sampled
        ######################### X local

        #### fcgf
        if self.isfcgf :
            globaldesc = self.compute_global_fcgf(outs, freeze_global=freeze_global)
            globaldesc_l2normed = tf.nn.l2_normalize(globaldesc, dim=-1, epsilon=1e-8, name='globaldesc')
            outs['global_desc'] = globaldesc_l2normed
        else :
            #### get global features
            if self.config.extract_global:
                globaldesc = self.compute_global_fcgf_dh3d(outs, freeze_global=freeze_global)
                globaldesc_l2normed = tf.nn.l2_normalize(globaldesc, dim=-1, epsilon=1e-8, name='globaldesc')
                outs['global_desc'] = globaldesc_l2normed

        ### loss
        if self.training:
            return self.compute_loss(outs)

    def compute_loss(self, outs):
        loss = 0.0

        # global loss
        if self.config.extract_global:
            global_loss = getattr(losses, self.global_loss_func)(global_descs=outs['global_desc'], **self.config)
            global_loss = tf.multiply(global_loss, self.config.global_loss_weight, name='globaldesc_loss')
            add_moving_summary(global_loss)
            loss += global_loss

        # local loss
        if self.config.add_local_loss:
            local_loss = getattr(losses, self.local_loss_func)(outs, **self.config)
            local_loss = tf.multiply(local_loss, self.config.local_loss_weight, name='localdesc_loss')
            add_moving_summary(local_loss)
            loss += local_loss

        ## detection loss
        if self.config.detection and self.config.add_det_loss:
            det_loss = getattr(losses, self.detection_loss_func)(outs, **self.config)
            det_loss = tf.multiply(det_loss, self.config.det_loss_weight, name='det_loss')
            add_moving_summary(det_loss)
            loss += det_loss

        loss = tf.identity(loss, name="gl_loc_loss")
        add_moving_summary(loss)

        if self.config.add_weight_decay:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(self.config.train_weight_decay), name='wd_cost')
        else:
            wd_cost = 0
        total_cost = tf.add(wd_cost, loss, name='total_cost')
        add_moving_summary(total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.config.start_lr,
            global_step=get_global_step_var(),
            decay_steps=self.config.decay_step,
            decay_rate=self.config.decay_rate, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)
