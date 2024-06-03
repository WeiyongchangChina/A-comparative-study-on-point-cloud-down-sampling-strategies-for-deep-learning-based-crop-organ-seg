#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:55:37 2019

@author: dell
"""

#import sys
#reload(sys) 
#sys.setdefaultencoding('utf8')
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
        # Point Feature Embedding output(B N NSAMLE 64)
        for i, num_out_channel in enumerate(mlp):
            new_points0 = tf_util.conv2d(new_points0, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv0%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv1%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
            new_points2 = tf_util.conv2d(new_points2, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv2%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        # Pooling in Local Regions
        new_points0 = tf.reduce_max(new_points0, axis=[2], keepdims=True, name='maxpool')
        new_points0 = tf.squeeze(new_points0, [2]) # (batch_size, npoints, 64)
        new_points1 = tf.reduce_max(new_points1, axis=[2], keepdims=True, name='maxpool')
        new_points1 = tf.squeeze(new_points1, [2]) # (batch_size, npoints, 64)
        new_points2 = tf.reduce_max(new_points2, axis=[2], keepdims=True, name='maxpool')
        new_points2 = tf.squeeze(new_points2, [2]) # (batch_size, npoints, 64)
        # connect difference scale point
        new_points = tf.concat([new_points0,new_points1],axis = -1)
        new_points = tf.concat([new_points,new_points2],axis = -1)
        return new_points #(batch_size, npoints, 96)
    
def feature_nearst(feature_data,points_xyz,nsample_point,mlp1,mlp2,mlp3,scope,is_training,bn_decay,radius=None,bn=True):
    feature_points = feature_data
    point_size = feature_points.get_shape()[1].value
    feature_points = tf.expand_dims(feature_points,2)#(batch_size, npoints, 1, 96)
    with tf.variable_scope(scope) as sc:
        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            feature_points = tf_util.conv2d(feature_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv3%d'%(i), bn_decay=bn_decay)
        # pooling in globle feature
        feature_points1 = tf.squeeze(feature_points,[2])
        net_globle = tf.reduce_max(feature_points, axis=[1], keepdims=True, name='maxpool2')#Bx1x1x64
        net_globle = tf.squeeze(net_globle, [2])#Bx1x64
        net_globle = tf.tile(net_globle, [1, point_size, 1])
        # POINT FEATURE EMBEDDING for finding nearst feature points
        for i, num_out_channel1 in enumerate(mlp2):
            feature_points = tf_util.conv2d(feature_points, num_out_channel1, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv4%d'%(i), bn_decay=bn_decay)
        
        feature_points = tf.squeeze(feature_points, [2])#(batch_size, npoints, 3)
        #Grouping
        net_group = sample_and_group(radius, nsample_point, feature_points, feature_points1, knn=True, use_xyz=False)#B X N x 64 x 64
        #point feature embedding
        for j, num_out_channel2 in enumerate(mlp3):
            net_group = tf_util.conv2d(net_group, num_out_channel2, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv5%d'%(j), bn_decay=bn_decay) 
        # Pooling in Local Regions
        net_group = tf.reduce_max(net_group, axis=[2], keepdims=True, name='maxpool')#B X N x 1 x 32
        net_group = tf.squeeze(net_group, [2])#B X N x 32
        # concta
        net_group = tf.concat(axis=-1, values=[feature_data, net_group])
        net_group = tf.concat(axis=-1, values=[net_group, net_globle])#128
        return feature_points, net_group
