#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:21:23 2019

@author: sgl
"""
import numpy as np
import glob
import os
import sys
import tensorflow as tf
import tf_util
#from pointnet_util import pointnet_sa_module, pointnet_fp_module
#from loss import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# -----------------------------------------------------------------------------
# PREPARE multyscale DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_idx(scale_idx, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    data = scale_idx[0];
    N = len(scale_idx[0])
    if (N == num_sample):
        return data
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample,...]
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample,...]
        return np.concatenate([data, dup_data], 0)


def mult_scale_point_model(input_data_lable,k_nearest_number,point_scale):
    """ find multyscale points in data_lable.
    Args:
        input_data_lable: B x N x F numpy array
        k_nearest_number:int, how many points to find around every points.
        point_scale: 1 x 3 array, represent the small, middle and large radious
    Returns:
        output_data_lable: B x N x F np array
    """
    # find the k nearest points
    point_xyz = input_data_lable[:,:,0:3]
    adj_matrix = tf_util.pairwise_distance(point_xyz)#BXNXN distance of every point to others points

    fusion_feature_point = []
    for idx in range(input_data_lable.shape[0]):#batch_size 
        mult_scale_point = []
        adj_matrix_idx = adj_matrix[idx,...]
        for idx1 in range(adj_matrix_idx.shape[0]):
            scale1_idx = np.where(adj_matrix_idx[idx1,:]<point_scale[0])
            scale2_idx = np.where(adj_matrix_idx[idx1,:]<point_scale[1])
            scale3_idx = np.where(adj_matrix_idx[idx1,:]<point_scale[2])
            single_mult_scale = np.stack((input_data_lable[idx,sample_idx(scale1_idx, k_nearest_number),:],
                                                           input_data_lable[idx,sample_idx(scale2_idx, k_nearest_number),:],
                                                           input_data_lable[idx,sample_idx(scale3_idx, k_nearest_number),:]),axis=0)
            mult_scale_point.append(single_mult_scale)
        mult_scale_point = np.stack(mult_scale_point[:],axis=0)
        mult_scale_point = np.transpose(mult_scale_point,axes=(0, 2, 3, 1))
        fusion_feature_point.append(mult_scale_point)
    fusion_feature_point = np.stack(fusion_feature_point[:],axis=0)
    fusion_feature_point = np.reshape(fusion_feature_point,(input_data_lable.shape[0]*input_data_lable.shape[1],k_nearest_number,input_data_lable.shape[2],3))
    return fusion_feature_point
# =============================================================================
#     # Point features (MLP implemented as conv2d)
#     input_point = tf.cast(fusion_feature_point, tf.float32)
#     net = tf_util.conv2d(input_point, 32, [1,input_data_lable.shape[2]],
#                          padding='VALID', stride=[1,1],
#                          bn=True, is_training=tf.cast(True, tf.bool),
#                          scope='conv1', bn_decay=None)
#     net = tf_util.conv2d(net, 64, [1,1],
#                          padding='VALID', stride=[1,1],
#                          bn=True, is_training=tf.cast(True, tf.bool),
#                          scope='conv2', bn_decay=None)
#     net = tf.reduce_max(net, axis=1, keep_dims=True)
#     net = tf.squeeze(net)
#     net = tf.reshape(net,(input_data_lable.shape[0],input_data_lable.shape[1],-1))
#     
# =============================================================================
    #    

#if __name__ == "__main__":
#    num_sample = 30
#    scale = [0.05,0.10,0.15]
#    all_data_lable = np.load("Area_1_conferenceRoom_1.npy")
#    batch_size = 10
#    num_point = 4096
#    data_lable_list = []
#    for i in range(batch_size):
#        data_lable_list.append(all_data_lable[i*num_point:(i+1)*num_point,:])
#    #data_lable = tf.transpose(np.dstack(data_lable_list),[2,0,1]).astype(np.uint8)#B X N X F  24x4096x8
#    temp_lable = np.concatenate(data_lable_list, 0)
#    data_lable= np.zeros((batch_size, num_point, 8))
#    for index in range(batch_size):
#        data_lable[index,...] = temp_lable[index*num_point:(index+1)*num_point,:]
#    add_model(data_lable,num_sample,scale)
#    tf.reset_default_graph()