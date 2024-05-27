#import open3d as o3d
import numpy as np
import os
import torch
from os import listdir, path

#@numba.jit()
def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
             Filelist.append(os.path.join(home, filename))
    return Filelist

def get_files(path):
    for home, dirs, files in os.walk(path):
        return files


def farthest_point_sample(xyz, npoint, z):
    device = xyz.device
    N, C = xyz.shape
    torch.manual_seed(z)
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N, dtype=torch.float64).to(device) * 1e10
    farthest = torch.randint(0, N, (1,),dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def main():
    #Your files path
    DATA_FILES = get_filelist(path=r'...')
    path_str = r'...'  # 注意，此处路径不要有中文

    # Your files' name path
    FILES = get_files(path=r'...')

    i = 0
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" + str(z) + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for txt in DATA_FILES:
        # 去除pcd文件头
        with open(os.path.join(path_str, txt), 'r') as f:
            lines = f.readlines()
            lines = lines[13:]
        points = np.loadtxt(lines)
        ply_array = np.asarray(points)
        sample_count = 4096
        s = len(points)
        if s < 4096:
            ply_array = np.tile(ply_array, (2, 1))

        b = torch.from_numpy(ply_array)
        b = b[ : , : 3]
        #expand 10
        for z in range(0, 9):
            sampled_points_index = farthest_point_sample(b, sample_count, z)
            # save path
            np.savetxt(r'...' + str(FILES[i]) + str(z) + ".txt", ply_array[sampled_points_index], fmt="%f %f %f %d %d")
        i = i + 1

if __name__ == '__main__':
    main()