import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import random

import os
import sys
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#去掉文件名，返回目录#当前目录file的绝对路径
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)#对于模块和自己写的程序不在同一个目录下，可以把模块的路径通过sys.path.append(路径)添加到程序中，
#sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import tf_util
from model import *
 
parser = argparse.ArgumentParser()#命令行借口
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')#添加参数
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--input_list_train', type=str, default='/media/dawei-server/DATA - HDD8TB/wyc/ForNothing/PlantNet_pt/data/train_file_list.txt', help='Input data list file')
parser.add_argument('--input_list_test', type=str, default='/media/dawei-server/DATA - HDD8TB/wyc/ForNothing/PlantNet_pt/data/test_file_list.txt', help='Input data list file')
parser.add_argument('--restore_model', type=str, default='log_plant_rseal_obj_01/', help='Pretrained model')
FLAGS = parser.parse_args()#解析参数
EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
TRAINING_FILE_LIST = FLAGS.input_list_train
TEST_FILE_LIST = FLAGS.input_list_test
PRETRAINED_MODEL_PATH = FLAGS.restore_model

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp pointnet_util.py %s' % (LOG_DIR)) # bkp of pointnet_util procedure

# os.system()将字符串转化成命令再服务器上运行
# system函数执行时，创建一个子进程在系统上执行命令行，子进程的执行结果无法影响主进程
# os.system('cd /usr/local && mkdir aaa.txt')在同一子进程,可执行并行操作
# cp命令用于复制文件或者目录 cp -r test/ newtest 将test/下的所有文件复制到新目录下

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
#记录训练日志文件log_train.txt。以及备份 model.py 和 train.py。


#全局变量。每个样本（此处每个样本为Block块，点数为4096，一共6个语义分类标签。
MAX_NUM_POINT = 4096
NUM_CLASSES = 6

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = DECAY_RATE
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
#BN_ 开头的4个变量用来计算 Batch Normalization 的Decay参数，即decay参数也随着训练逐渐decay
HOSTNAME = socket.gethostname()


# Load train data
train_file_list = provider.getDataFiles(os.path.join(ROOT_DIR,TRAINING_FILE_LIST))
train_data = []
train_group = []
train_sem = []
train_obj = []
for h5_filename in train_file_list:
    cur_data, cur_group, cur_sem, cur_obj = provider.load_h5_data_label_seg(os.path.join(ROOT_DIR,h5_filename))
    train_data.append(cur_data)
    train_group.append(cur_group)
    train_sem.append(cur_sem)
    train_obj.append(cur_obj)
train_data = np.concatenate(train_data, axis=0)#能够一次完成多个数组的拼接。
train_group = np.concatenate(train_group, axis=0)
train_sem = np.concatenate(train_sem, axis=0)
train_obj = np.concatenate(train_obj,axis=0) 

# Load test data
test_file_list = provider.getDataFiles(os.path.join(ROOT_DIR,TEST_FILE_LIST))
test_data = []
test_group = []
test_sem = []
test_obj = []
for h5_filename_test in test_file_list:
    cur_data, cur_group, cur_sem, cur_obj = provider.load_h5_data_label_seg(os.path.join(ROOT_DIR,h5_filename_test))
    test_data.append(cur_data)
    test_group.append(cur_group)
    test_sem.append(cur_sem)
    test_obj.append(cur_obj)
test_data = np.concatenate(test_data,axis=0)
test_group = np.concatenate(test_group,axis=0)
test_sem = np.concatenate(test_sem,axis=0)
test_obj = np.concatenate(test_obj,axis=0)


def log_string(out_str):  #log_string(out_str)函数用来log训练日志。
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):#算指数衰减的学习率。训练时学习率最好随着训练衰减
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!  根0。00001比较，取大值
    return learning_rate        

def get_bn_decay(batch):   #计算衰减的Batch Normalization 的 decay。基本同上。
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_trainable_variables():
    #trainables = [var for var in tf.trainable_variables() if 'bias' not in var.name]# and \
    trainables = tf.trainable_variables()#这个函数可以也仅可以查看可训练的变量
    print("All {} trainable variables, {} variables to train".format(len(tf.trainable_variables()), len(trainables)))
    return trainables


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):#，可以指定运行在GPU还是CUP上，以及哪块GPU上。

            pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)#创建3tensor
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # add by Shi Guoliang 20201219
            ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph = \
                placeholder_onehot_inputs(BATCH_SIZE, NUM_POINT)##创建4tensor
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)#tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称，用法如下：
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)#用来显示标量信息，其格式为：一般在画loss,accuary时会用到这个函数。

            # Get model and loss 
            pred_sem, pred_ins, fuse_catch = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            pred_sem_softmax = tf.nn.softmax(pred_sem)#softmax函数的作用就是归一化
            pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)#tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引

            loss, sem_loss, disc_loss, l_var, l_dist, l_reg, simmat_loss = \
                get_loss(pred_ins, labels_pl, pred_sem, sem_labels_pl, fuse_catch, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph)
            
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('sem_loss', sem_loss)
            tf.summary.scalar('disc_loss', disc_loss)
            tf.summary.scalar('l_var', l_var)
            tf.summary.scalar('l_dist', l_dist)
            tf.summary.scalar('l_reg', l_reg)
            tf.summary.scalar('simmat_loss', simmat_loss)

            
            trainables = get_trainable_variables()
    
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)#优化类提供了为损失函数计算梯度的方法
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            
            train_op = optimizer.minimize(loss, var_list=trainables, global_step=batch)
           
             
            load_var_list = [v for v in tf.all_variables() if ('sem_' not in v.name)]
            loader = tf.train.Saver(load_var_list, sharded=True)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=MAX_EPOCH)
            
            
        # Create a session
        config = tf.ConfigProto()   #配置session 运行参数。
        config.gpu_options.allow_growth = True #让TensorFlow在运行过程中动态申请显存，避免过多的显存占用
        config.allow_soft_placement = True   #当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行。
        config.log_device_placement = True   #在终端打印出各项操作是在哪个设备上运行的
        sess = tf.Session(config=config)

        # Add summary writers
   #merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。     
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),#指定一个文件用来保存图。
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            log_string("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            log_string("Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        sess.run(adam_initializers)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'ptsseglabel_ph':ptsseglabel_ph,
               'ptsgroup_label_ph':ptsgroup_label_ph,
               'pts_seglabel_mask_ph':pts_seglabel_mask_ph,
               'pts_group_mask_ph':pts_group_mask_ph,
               'pred_sem':pred_sem,
               'pred_ins':pred_ins,
               'loss': loss,
               'sem_loss': sem_loss,
               'disc_loss': disc_loss,
               'l_var': l_var,
               'l_dist': l_dist,
               'l_reg': l_reg,
               'simmat_loss': simmat_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer) #train_one_epoch 函数用来训练一个epoch，eval_one_epoch函数用来每运行一个epoch后evaluate在测试集的accuracy和loss。每10个epoch保存1次模型。

            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 2 == 0 or epoch == (MAX_EPOCH - 1):
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, shuffled_idx = provider.shuffle_data(train_data[:, :, :], train_group)
    current_sem = train_sem[shuffled_idx].astype(int)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    loss_sum = 0
    
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for batch_idx in range(num_batches):
        if batch_idx % 50 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        pts_label_one_hot, pts_label_mask = convert_seg_to_one_hot(current_sem[start_idx:end_idx])
        pts_group_label, pts_group_mask = convert_groupandcate_to_one_hot(current_label[start_idx:end_idx])
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['sem_labels_pl']: current_sem[start_idx:end_idx],
                     ops['ptsseglabel_ph']: pts_label_one_hot,
                     ops['pts_seglabel_mask_ph']: pts_label_mask,
                     ops['ptsgroup_label_ph']: pts_group_label,
                     ops['pts_group_mask_ph']: pts_group_mask,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, sem_loss_val, disc_loss_val, l_var_val, l_dist_val, l_reg_val, l_simmat_loss, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['sem_loss'], ops['disc_loss'], ops['l_var'], ops['l_dist'], ops['l_reg'], ops['simmat_loss'], ops['pred_sem']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) #B X N 
        correct = np.sum(pred_val == current_sem[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_sem[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
        
        if batch_idx % 50 == 0:
            log_string("loss: {:.2f}; sem_loss: {:.2f}; disc_loss: {:.2f}; l_var: {:.2f}; l_dist: {:.2f}; l_reg: {:.3f}; simmat_loss: {:.3f}.".format(loss_val, sem_loss_val, disc_loss_val, l_var_val, l_dist_val, l_reg_val, l_simmat_loss))
    
    log_string('train mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('train accuracy: %f'% (total_correct / float(total_seen)))
    log_string('train avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


# def eval_one_epoch(sess, ops, test_writer):
#     """ ops: dict mapping from string to tf ops """
#     is_training = False
#     total_correct = 0
#     total_seen = 0
#     loss_sum = 0
#     total_seen_class = [0 for _ in range(NUM_CLASSES)]
#     total_correct_class = [0 for _ in range(NUM_CLASSES)]
#
#     log_string('----')
#     current_data = test_data
#     current_label = np.squeeze(test_group)  ## 从数组的形状中删除单维条目，即把shape中为1的维度去掉
#     current_sem = np.squeeze(test_sem).astype(np.uint8)
#
#     file_size = current_data.shape[0]
#     num_batches = file_size // BATCH_SIZE
#     for batch_idx in range(num_batches+1):
#         if batch_idx == num_batches:
#             start_idx = file_size-BATCH_SIZE
#             end_idx = file_size
#         else:
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = (batch_idx+1) * BATCH_SIZE
#
#         pts_label_one_hot, pts_label_mask = convert_seg_to_one_hot(current_sem[start_idx:end_idx])
#         pts_group_label, pts_group_mask = convert_groupandcate_to_one_hot(current_label[start_idx:end_idx])
#
#         feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
#                      ops['labels_pl']: current_label[start_idx:end_idx],
#                      ops['sem_labels_pl']: current_sem[start_idx:end_idx],
#                      ops['ptsseglabel_ph']: pts_label_one_hot,
#                      ops['pts_seglabel_mask_ph']: pts_label_mask,
#                      ops['ptsgroup_label_ph']: pts_group_label,
#                      ops['pts_group_mask_ph']: pts_group_mask,
#                      ops['is_training_pl']: is_training,}
#
#         summary, step, loss_val, pred_val_all = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred_sem']],
#                                       feed_dict=feed_dict)
#         test_writer.add_summary(summary, step)
#         # save pred value
#         pred_val = np.argmax(pred_val_all, 2) # B X N
#         correct = np.sum(pred_val == current_sem[start_idx:end_idx])
#         total_correct += correct
#         total_seen += (BATCH_SIZE*NUM_POINT)
#         loss_sum += (loss_val*BATCH_SIZE)
#         for i in range(start_idx, end_idx):
#             for j in range(NUM_POINT):
#                 l = current_sem[i, j]
#                 total_seen_class[l] += 1
#                 total_correct_class[l] += (pred_val[i-start_idx, j] == l)
#
#     log_string('test mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
#     log_string('test accuracy: %f'% (total_correct / float(total_seen)))
#     log_string('test avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    num_batches = (test_data.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

    loss_sum_myself = 0
    total_seen_myself = 0
    total_correct_myself = 0
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data
    current_label = np.squeeze(test_group)  ## 从数组的形状中删除单维条目，即把shape中为1的维度去掉
    current_sem = np.squeeze(test_sem).astype(np.uint8)

    seg_classes = {'plant1': [0, 1], 'plant2': [2, 3], 'plant3': [4, 5]}
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    for batch_idx in range(num_batches + 1):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(test_data.shape[0], (batch_idx + 1) * BATCH_SIZE)
        cur_batch_size = end_idx - start_idx
        cur_batch_data = current_data[start_idx:end_idx, :, :]
        cur_batch_label = current_sem[start_idx:end_idx]

        if batch_idx == num_batches:
            start_idx = file_size - BATCH_SIZE
            end_idx = file_size
        else:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
        # ---------------------------------------------------------------------

        pts_label_one_hot, pts_label_mask = convert_seg_to_one_hot(current_sem[start_idx:end_idx])
        pts_group_label, pts_group_mask = convert_groupandcate_to_one_hot(current_label[start_idx:end_idx])

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['sem_labels_pl']: current_sem[start_idx:end_idx],
                     ops['ptsseglabel_ph']: pts_label_one_hot,
                     ops['pts_seglabel_mask_ph']: pts_label_mask,
                     ops['ptsgroup_label_ph']: pts_group_label,
                     ops['pts_group_mask_ph']: pts_group_mask,
                     ops['is_training_pl']: is_training, }

        summary, step, loss_val, pred_val_all = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred_sem']],
                                                         feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # ---------------------------------------------------------------------
        cur_pred_val = pred_val_all[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        correct_myself = np.sum(cur_pred_val == cur_batch_label)
        total_correct_myself += correct_myself
        total_seen_myself += (cur_batch_size*NUM_POINT)
        if cur_batch_size==BATCH_SIZE:
            loss_sum_myself += loss_val

        # save pred value
        pred_val = np.argmax(pred_val_all, 2)  # B X N
        correct = np.sum(pred_val == current_sem[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_sem[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx, j] == l)

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = cur_batch_label[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_shape_ious = sum(shape_ious.values()) / len(shape_ious)
    # log_string('eval mean loss: %f' % (loss_sum_myself / float(test_data.shape[0] / BATCH_SIZE)))
    # log_string('eval accuracy: %f' % (total_correct_myself / float(total_seen_myself)))
    # log_string('eval avg class acc: %f' % (
    #     np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))
    EPOCH_CNT += 1

    log_string('test mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
    log_string('test accuracy: %f' % (total_correct / float(total_seen)))
    log_string('test avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
