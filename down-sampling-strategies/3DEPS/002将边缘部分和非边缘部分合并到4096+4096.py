"""
Created on MON SEB 23 16:21:12 2022

@author: YC-W
"""
import os
from os import listdir, path
import numpy as np
import random

# 分别为内部点和边缘点的路径，以及合并后文件的保存路径
path_center = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\shimei_c'  # your directory path
path_edge = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\shimei_e'
save_path = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\shimei_hebing'

txt_cs = [f for f in listdir(path_center)
        if f.endswith('.txt') and path.isfile(path.join(path_center, f))]

txt_es = [f for f in listdir(path_edge)
        if f.endswith('.txt') and path.isfile(path.join(path_edge, f))]

#随机种子的选择
i = 0

for txt_c, txt_e in zip(txt_cs, txt_es):
        #暂时存储打乱顺序后的两部分点云
        c_temp = []
        e_temp = []
        end_temp = []
        i = i + 1
        with open(os.path.join(path_center, txt_c), 'r') as f:
                index = []
                lines = f.readlines()
                #得到点云点数
                size_c = len(lines)
                #随机取出4096个点
                np.random.seed(i)
                indexs = np.random.randint(0, size_c, 4096, int)
                #random.shuffle(lines)
                #print(lines)
                for index in indexs:
                        c_temp.append(lines[index])
                #print(lines)
        with open(os.path.join(path_edge, txt_e), 'r') as f:
                index = []
                lines = f.readlines()
                size_e = len(lines)
                np.random.seed(i)
                indexs = np.random.randint(0, size_e, 4096, int)
                #random.shuffle(lines)
                for index in indexs:
                        e_temp.append(lines[index])

        end_temp = c_temp + e_temp
        #print(end_temp)

        #保存
        with open(os.path.join(save_path, os.path.splitext(txt_e)[0] + "end.txt"), 'w') as f:
                f.write(''.join(end_temp[0:]))