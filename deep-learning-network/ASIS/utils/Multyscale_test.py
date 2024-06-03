#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:21:23 2019

@author: sgl
"""
import sys
reload(sys) 
sys.setdefaultencoding('utf8')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import tf_util
# -----------------------------------------------------------------------------
# PREPARE multyscale DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_and_group(radius, nsample, xyz, points, knn, use_xyz):
    '''
    Input:
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    if knn:
        _,idx = knn_point(nsample, xyz, xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1,1,nsample,1]) # translation normalization ??????
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_points

def add_model(input_data, input_feature, nsample_number, point_scale, mlp, is_training, bn_decay, scope, bn=True, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Grouping
        new_points0 = sample_and_group(point_scale[0], nsample_number,
                                      input_data, input_feature, knn=False, use_xyz=True) #(batch_size, npoint, nsample_number, 3+channel)
        new_points1 = sample_and_group(point_scale[1], nsample_number,
                                      input_data, input_feature, knn=False, use_xyz=True) #(batch_size, npoint, nsample_number, 3+channel)
        new_points2 = sample_and_group(point_scale[2], nsample_number,
                                      input_data, input_feature, knn=False, use_xyz=True) #(batch_size, npoint, nsample_number, 3+channel)
        new_points3 = tf.concat([new_points0,new_points1],axis = -1)
        #(batch_size, npoint, nsample_number, 3*(3+channel))
        new_points = tf.concat([new_points3,new_points2],axis = -1)
        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='cbonv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        # Pooling in Local Regions
        new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, 128)
        return tf.concat([input_data, new_points], axis=-1)#(batch_size, npoints, 128+3)
    
# mlp=[32,64,128] mlp1=[256,128,64] mlp2=[128,128,64]
def feature_nearst(feature_data,points_xyz,nsample_point,mlp1,mlp2,scope,is_training,bn_decay,radius=None,bn=True):
    feature_points = feature_data
    feature_points = tf.expand_dims(feature_points,2)#(batch_size, npoints, 1, 128+3)
    with tf.variable_scope(scope) as sc:
        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            feature_points = tf_util.conv2d(feature_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay)
        # full connect
        feature_points = tf.squeeze(feature_points, [2])#(batch_size, npoints,64)
        net_ins1 = tf_util.conv1d(feature_points, 32, 1, padding='VALID', bn=True, is_training=is_training, scope='first_feature1', bn_decay=bn_decay)
        net_ins1 = tf_util.dropout(net_ins1, keep_prob=0.5, is_training=is_training, scope='first_feature2')
        net_ins1 = tf_util.conv1d(net_ins1, 3, 1, padding='VALID', activation_fn=None, scope='first_feature3')#B X N X 3
        # pooling in globle feature
        net_globle = tf.reduce_max(net_ins1, axis=[1], keep_dims=True, name='maxpool2')#Bx1x5
        # concat
        net_globle = tf.tile(net_globle, [1, 4096, 1])
        net_globle_concat0 = tf.concat(axis=-1, values=[feature_data, feature_points])# (batch_size, npoints, 128+3+64)
        net_globle_concat1 = tf.concat(axis=-1, values=[net_globle_concat0, net_globle])# (batch_size, npoints, 3+128+64+5)
        #Grouping
        net_group = sample_and_group(radius, nsample_point, net_ins1, net_globle_concat1, knn=True, use_xyz=False)#B X N x 32 x 200
        #point feature embedding
        for j, num_out_channel2 in enumerate(mlp2):
            net_group = tf_util.conv2d(net_group, num_out_channel2, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='cdonv%d'%(j), bn_decay=bn_decay) 
        # Pooling in Local Regions
        net_group = tf.reduce_max(net_group, axis=[2], keep_dims=True, name='maxpool')#B X N x 1 x 64
        net_group = tf.squeeze(net_group, [2])#B X N x 64
#        return tf.concat([points_xyz, net_group], axis=-1)#B X N x 64+3
        return net_ins1, net_group
        
        