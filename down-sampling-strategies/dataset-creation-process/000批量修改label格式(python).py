# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:46:48 2019

@author: 426-4
"""

import os
from os import listdir, path

path_str = 'D:\cpp_project\PCL1\point_data'  # your directory path
txts = [f for f in listdir(path_str)
        if f.endswith('.pcd') and path.isfile(path.join(path_str, f))]

for txt in txts:
    with open(os.path.join(path_str, txt), 'r') as f:
        lines = f.readlines()
        lines[4] = lines[4].replace('I','U')  ##########4可以改为3，注意看pcd格式

    with open(os.path.join(path_str,os.path.splitext(txt)[0]+".pcd"), 'w') as f:
        f.write(''.join(lines[0:]))
