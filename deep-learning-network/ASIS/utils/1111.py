#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:06:16 2019

@author: sgl
"""

import tensorflow as tf
 
t1 = tf.Variable([[0.1,0.2,0.3],[0.4,1.5,2.6],[2.0,2.1,2.2]])
to_add = tf.range(t1.get_shape()[1])#[0 1 2 ... 4095]
to_add = tf.reshape(to_add, [-1, 1])#shape:4096 x 1
to_add = tf.tile(to_add, [1, 2]) #shape: 4096 x k

vals, nn_idx = tf.nn.top_k(t1, k=2)
mask = tf.cast(vals > 1*0.3, tf.int32)
idx_to_add = to_add * mask
nn_idx1 = nn_idx * (1 - mask) + idx_to_add 
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#    sess.run(t2)
#    print (t2)
    print(sess.run(vals))
    print(sess.run(nn_idx))
    print(sess.run(mask))
    print(sess.run(to_add))
    print(sess.run(idx_to_add))
    print(sess.run(nn_idx1))