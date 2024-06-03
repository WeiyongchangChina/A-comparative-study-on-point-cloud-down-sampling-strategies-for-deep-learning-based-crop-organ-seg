import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, is_dist=False, bn=True):
    data_format = 'NHWC'
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, :3]

    k = 20

    adj = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.dg_knn(adj, k=k)  # (batch, num_points, k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    out1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv1d', bn_decay=bn_decay,
                          data_format=data_format)

    out2 = tf_util.conv2d(out1, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv2d', bn_decay=bn_decay,
                          data_format=data_format)

    net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.dg_knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)
    
    out3 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv3d', bn_decay=bn_decay,
                          data_format=data_format)

    out4 = tf_util.conv2d(out3, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv4d', bn_decay=bn_decay,
                          data_format=data_format)

    net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.dg_knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)
    
    out5 = tf_util.conv2d(edge_feature, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv5d', bn_decay=bn_decay,
                          data_format=data_format)

    # out6 = tf_util.conv2d(out5, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, weight_decay=weight_decay,
    #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

    net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=bn, is_training=is_training,
                          scope='conv7d', bn_decay=bn_decay,
                          data_format=data_format)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1')
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv2')
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 6, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3')
    net = tf.squeeze(net, [2])

    return net, end_points



def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
