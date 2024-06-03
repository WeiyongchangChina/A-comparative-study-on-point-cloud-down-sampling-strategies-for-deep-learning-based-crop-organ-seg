import os
import shutil
import numpy as np
arr=np.load(r"D:\cpp_project\PCL1\downsampling_ex\FPS\test_index.npy")
print(arr.shape)#182
file_dir = r"D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\0.85"
for root, dirs, files in os.walk(file_dir, topdown=False):
    print(root)     # 当前目录路径
    print(dirs)     # 当前目录下所有子目录
    print(files)        # 当前路径下所有非目录子文件
print(len(files))#546
number=1820
filter=[0]*1820
k = 0
for i in range(182):
    a=arr[i]*10
    for j in range(10):
        a=files[arr[i]*10+j]
        filter[k]=a
        k = k + 1
print(filter)
dir_CRC=r"D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\0.85"
dir_CRC_2=r"D:\cpp_project\PCL1\downsampling_ex\3DEPS\3DEPS不同ratio\处理结果\0.85\res_test"
for i in filter:
    # 目录的拼接
        full_path = os.path.join(dir_CRC,i)
        # 移动文件
        shutil.move(full_path,dir_CRC_2)
