import os
import numpy as np
import h5py
import queue
import threading
import point_op
import pc_util
import random
import math
from configs import FLAGS
from glob import glob
import open3d as o3d

def poisson_r(point, sample_num):
    xmin = np.min(point[:, 0])
    ymin = np.min(point[:, 1])
    zmin = np.min(point[:, 2])
    xmax = np.max(point[:, 0])
    ymax = np.max(point[:, 1])
    zmax = np.max(point[:, 2])
    area = (xmax-xmin)*(ymax-ymin)+(xmax-xmin)*(zmax-zmin)+(ymax-ymin)*(zmax-zmin)
    r = np.sqrt(area/(0.7*math.pi*sample_num))
    return r

def poisson_num(point,r):
    sample = []
    qualified = point
    while(qualified.shape[0]!=0):
        i = np.random.randint(0, qualified.shape[0], dtype=np.int64)
        sample_point = qualified[i,:]
        sample.append(sample_point)
        dis = np.sum((qualified - sample_point)**2,axis=-1)
        qualify = []
        for j in range(qualified.shape[0]):
            if dis[j] > r**2:
                qualify.append(point[j])
        qualified = np.array(qualify)
    sample = np.array(sample)
    num = sample.shape[0]
    return sample, num

    # q = queue.Queue()
    # q.put(sample_point)
    # while(q.empty() == False):

def poisson_sampling(point, sample_num):
    r = poisson_r(point, sample_num)
    # cell = r/np.sqrt(3)
    sample, num = poisson_num(point, r)
    while(num < sample_num):
        rmin = r / 2.0
        sample, num = poisson_num(point, rmin)
        rmax = r
        r = rmin
        # print(num)
    while (num > sample_num):
        rmax = r * 2.0
        sample, num = poisson_num(point, rmax)
        rmin = r
        r = rmax
        # print(num)
        if (num < sample_num):
            break
    while(num != sample_num):
        r = (rmin+rmax)/2
        sample, num = poisson_num(point, r)
        if num < sample_num :
            rmax = r
        if num > sample_num:
            rmin = r
        # print(num,r)
    return sample

def patch(file,patch_num,patch_num_ratio):
    data = pc_util.read_ply(file)[:, :3]
    # data = pc_util.read_ply_with_color(file)
    seed_num = int(patch_num * 3)
    patch_num_point = int(data.shape[0]/patch_num)
    patch, idx = pc_util.downsample_points(data, seed_num)
    idx = idx[:seed_num]
    patches = pc_util.extract_knn_patch(data[np.asarray(idx), :], data, patch_num_point)
    new_patch = np.zeros((patches.shape[0], 1024, patches.shape[2]),dtype=np.float32)
    for i in range(patches.shape[0]):
        # id = point_op.nonuniform_sampling(patches.shape[1], sample_num=1024)  # 非均匀采样
        sample = poisson_sampling(patches[i], sample_num=1024)
        new_patch[i, ...] = sample
        print(i)
    return new_patch

# DIR = 'C:/Users/15463/Desktop/1/venv/pu_gan2.0/data/train/building/ply'
# save_dir = 'C:/Users/15463/Desktop/1/venv/pu_gan2.0/data/train/building/npy'
# samples = glob(os.path.join(DIR,'*.ply'))
# for point_path in samples:
#     print(point_path)
#     save = os.path.join(save_dir, point_path.split('\\', 2)[-1][:-4] + '.npy')
#     data = patch(point_path,80,3)
#     np.save(save, data)
# file = os.path.join(FLAGS.data_dir, 'train/pu-net/block.ply')
# data = patch(file,1,3)
# save = os.path.join(FLAGS.data_dir, 'train/block1024.npy')
# np.save(save, data)

file = os.path.join(FLAGS.data_dir, 'train/building/npy')
d = None
for i in os.listdir(file):
    data = np.load(file + '/' + i)
    # if d is not None:
    #     d = np.concatenate((d,data),axis=0)
    # else:
    #     d = data
    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors = o3d.utility.Vector3dVector(np.zeros((204800,3)))
    o3d.visualization.draw_geometries([pt1], width=50000, height=50000)
    print(i)
save = os.path.join(FLAGS.data_dir, 'train/building.npy')
np.save(save, d)