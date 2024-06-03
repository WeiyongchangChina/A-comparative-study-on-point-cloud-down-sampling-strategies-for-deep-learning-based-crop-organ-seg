# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:36:12 2021

@author: JS-L
"""

import numpy as np
import os

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

DATA_FILES =get_filelist(path=r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\去除背景点和噪声edge')
for i in range(len(DATA_FILES)):
    temp=np.loadtxt(DATA_FILES[i])
    shape=temp.shape
    temp[:,3]=temp[:,3]-2#标签减二
    zeros=np.zeros((shape[0],shape[1]+1))
    #print(zeros)
    zeros[:,:4]=temp[:,:4]
    #print(zeros)
    print(DATA_FILES[i][70:71])
    if DATA_FILES[i][70:71]=="b" :#添加object
        zeros[:,4]=0
    if DATA_FILES[i][70:71]=="m" :
        zeros[:,4]=1
    if DATA_FILES[i][70:71]=="s":
         zeros[:,4]=2
    np.savetxt(r"D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\改标签edge\\"+str(DATA_FILES[i][70:]),zeros,fmt="%f %f %f %d %d")
    
    
    