# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:46:48 2019

@author: 426-4
"""

import os
from os import listdir, path

path_str = r'D:\cpp_project\PCL1\downsampling_ex\RS\rs_seed(3407)'  # your directory path
# txts = [f for f in listdir(path_str)
#         if f.endswith('c.pcd') and path.isfile(path.join(path_str, f))] #此处要换一下，b,c
txts = [f for f in listdir(path_str)
        if f.endswith('.txt') and path.isfile(path.join(path_str, f))]
save_path_str = r'D:\cpp_project\PCL1\downsampling_ex\RS\rs_seed再次去头' #此处要换一下，b,c
for txt in txts:
    with open(os.path.join(path_str, txt), 'r') as f:
        lines = f.readlines()

    with open(os.path.join(save_path_str,os.path.splitext(txt)[0]+".txt"), 'w') as f:
        f.write(''.join(lines[11:]))