#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:53:03 2019

@author: sgl
"""
import os
import sys
import numpy as np
import h5py


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:,0:3]
    ins_label = (data[:,3]).astype(int)
    find_index = np.where(ins_label>=1)
    sem_label = np.zeros((data.shape[0]), dtype=int)
    #obj_lable = data[:,4]
    
    #sem_label[find_index] = 1
    #print(np.sum(sem_label),len(data))
    
    return point_xyz, ins_label, sem_label

def change_scale(data):#这一步的主要目的是将点云中心移动到坐标原点，并将所有点的坐标的绝对值限制在1以内
    # #centre 
    # xyz_min = np.min(data[:,0:3],axis=0)
    # xyz_max = np.max(data[:,0:3],axis=0)
    # data[:,0:3] = 2*(data[:,0:3]-xyz_min)/(xyz_max - xyz_min) - 1
    # return data[:,0:3]
    ####################上面时陈迎亮有毒版本
    xyz_min = np.min(data[:,0:3],axis=0)
    xyz_max = np.max(data[:,0:3],axis=0)
    xyz_move = xyz_min+(xyz_max-xyz_min)/2
    data[:,0:3] = data[:,0:3]-xyz_move
    #scale
    scale = np.max(data[:,0:3])
    #change_data[:,0:3] = data[:,0:3]/scale
    return data[:,0:3]/scale

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

if __name__ == "__main__":
    DATA_FILES =get_filelist(path=r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\处理结果\0.85\res_train')
    num_sample = 4096
    DATA_ALL = []

    for fn in range(len(DATA_FILES)):
        #print(DATA_FILES[fn])
        current_data, current_ins_label, current_sem_label = loadDataFile(DATA_FILES[fn])
        #print(len(current_ins_label))
        for i in range(len(current_ins_label)):
            print(DATA_FILES[fn][75])
            #print(DATA_FILES[fn][32])#37
            #print(current_ins_label)
            if(DATA_FILES[fn][75]=='b' and current_ins_label[i]>=1):
                current_sem_label[i]=1
            if(DATA_FILES[fn][75]=='b' and current_ins_label[i]==0):
                current_sem_label[i]=0
            if(DATA_FILES[fn][75]=='m' and current_ins_label[i]>=1):
                current_sem_label[i]=3
            if(DATA_FILES[fn][75]=='m' and current_ins_label[i]==0):
                current_sem_label[i]=2
            if(DATA_FILES[fn][75]=='s' and current_ins_label[i]>=1):
                current_sem_label[i]=5
            if(DATA_FILES[fn][75]=='s' and current_ins_label[i]==0):
                current_sem_label[i]=4 
        #print(current_sem_label)
        #print("yeh")     
        
        #print(current_sem_label.shape)
        change_data = change_scale(current_data)
#        data_sample,index = sample_data(change_data, num_sample)
        data_label = np.column_stack((change_data,current_ins_label,current_sem_label))
        DATA_ALL.append(data_label)
    print(np.asarray(DATA_ALL).shape)
    output = np.vstack(DATA_ALL)
    output = output.reshape(3640,num_sample,5)
    
    #output = np.asarray(DATA_ALL)
    print(output.shape)

    save_path = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\处理结果\0.85\3DEPS_4096_train0.85.h5'
            
    if not os.path.exists(r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\处理结果\0.85\3DEPS_4096_train0.85.h5'):
        with h5py.File(save_path,'w') as f:
#            sample = np.random.choice(8192, 2048)
            f['data'] = output[:,:,0:3]
            f['pid'] = output[:,:,3]#实例标签
            f['seglabel'] = output[:,:,4]
            f['obj'] = np.zeros(output[:,:,4].shape)-1




