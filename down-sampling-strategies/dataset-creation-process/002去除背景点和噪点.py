"""
Created on MON SEB 23 16:21:12 2022

@author: YC-W
"""
import os
from os import listdir, path


path_str = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\去头edge'  # your directory path
save_path_str = r'D:\cpp_project\PCL1\downsampling_ex\3DEPS\data_edge&core\去除背景点和噪声edge'

txts = [f for f in listdir(path_str)
        if f.endswith('.txt') and path.isfile(path.join(path_str, f))]
len_txts = len(txts) #判断文件夹下文件的数量

for txt in txts:
    with open(os.path.join(path_str, txt), 'r') as f:

        index = []
        lines = f.readlines()
        for line in lines:
            #print(line.split()[-2])     #读取空格分割的字符
            print(line.split()[0:3])
            #print(len(line))
            if(line.split()[-1] > '1' and line.split()[0:3] != ['0', '0', '0']):         #这里可以修改为需要的噪声的标签
                index.append(line)

    with open(os.path.join(save_path_str,os.path.splitext(txt)[0]+".txt"), 'w') as f:
        f.write(''.join(index[0:]))