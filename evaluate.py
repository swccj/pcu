import argparse
import os
import numpy as np
import tensorflow as tf
from glob import glob
import re
import csv
from collections import OrderedDict
import os
import pc_util
from pc_util import load, save_ply_property,get_pairwise_distance
from pc_util import normalize_point_cloud
from sklearn.neighbors import NearestNeighbors
import math
from time import time
from configs import FLAGS
import open3d as o3d

FLAGS.pred = os.path.join(FLAGS.data_dir,'pred')
FLAGS.gt = os.path.join(FLAGS.data_dir,'gt')
PRED_DIR = os.path.abspath(FLAGS.pred)
GT_DIR = os.path.abspath(FLAGS.gt)
print(PRED_DIR)
# NAME = FLAGS.name

print(GT_DIR)
gt_paths = glob(os.path.join(GT_DIR,'*.xyz'))

gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
print(len(gt_paths))

def square_distance(a, b):#确定每一个点距离采样点的距离

    B, N, _ = a.shape
    _, M, _ = b.shape
    dist = -2 * tf.matmul(a, tf.transpose(b,(0,2,1))) # 2*(xn * xm + yn * ym + zn * zm)
    # dist += np.sum(a ** 2, -1).resize(B, N, 1) # xn*xn + yn*yn + zn*zn
    a_ = tf.reduce_sum(a**2 , axis=-1,keepdims=True)
    dist = tf.add(dist , tf.reshape(a_,[B,N,1]))
    b_ = tf.reduce_sum(b**2 , axis=-1,keepdims=True)
    dist = tf.add(dist , tf.reshape(b_,[B,1,M]))
    return dist

def nn_distance(xyz1,xyz2):
    dist1 = square_distance(xyz1,xyz2)
    dist1 = tf.sort(dist1,axis=-1)
    dist1 = tf.slice(dist1, begin=[0, 0, 0], size=[xyz1.shape[0],xyz1.shape[1],1])
    dist2 = square_distance(xyz2, xyz1)
    dist2 = tf.sort(dist2, axis=-1)
    dist2 = tf.slice(dist2, begin=[0, 0, 0], size=[xyz1.shape[0],xyz1.shape[1],1])
    return dist1,dist2

def cal_nearest_distance(queries, pc, k=2):
    """
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis,knn_idx = knn_search.kneighbors(queries, return_distance=True)
    return dis[:,1]

def get_uniform_loss(pcd, percentages=[0.008,0.012], radius=1.0):
    B,N,C = pcd.shape
    npoint = int(N * 0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample
        pc = np.array(pcd)
        idx = op.farthest_point_sample(pc,npoint )
        new_xyz = np.zeros((B,npoint,3),dtype=np.float32) # (batch_size, npoint, 3)
        for i in range(B) :
            new_xyz[i,:,:] = pc[i,idx[i],:]# (batch_size, npoint, 3)
        # new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))
        # new_xyz = op.farthest_point_sample_(pcd, npoint)
        idx = op.query_ball_point(r, nsample, pcd, new_xyz, npoint)

        expect_len = tf.sqrt(disk_area)  # using square
        # grouped_pcd = np.zeros((B, npoint, nsample, 3), dtype=np.float32)
        grouped = None
        for i in range(B):
            # grouped_pcd[i, :, :] = pc[i, idx[i], :]
            grouped_pcd = tf.slice(pcd, begin=[i, 0, 0], size=[1, tf.shape(pcd)[1], tf.shape(pcd)[2]])
            idx_ =tf.slice(idx, begin=[i, 0, 0], size=[1, tf.shape(idx)[1], tf.shape(idx)[2]])
            grouped_pcd1 = op.index_points(grouped_pcd, idx_)
            if grouped is not None:
                grouped = tf.concat([grouped, grouped_pcd1], axis=0)
            else:
                grouped = grouped_pcd1
        grouped_pcd = tf.concat(tf.unstack(grouped, axis=1), axis=0)

        var, _ = get_feature.knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis+1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis,axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean*math.pow(p*100,2)
        #nothing 4
        loss.append(mean)
    return tf.add_n(loss)/len(percentages)

precentages = np.array([0.008, 0.012])
samples = glob(os.path.join(PRED_DIR,'*.xyz'))
for point_path in samples:
    pc = pc_util.load(point_path)[:, :3]
    name = point_path.split('\\', -1)[-1][:-4]
    gt_path = os.path.join(GT_DIR, name + '.xyz')
    gt = pc_util.load(gt_path)[:, :3]
    # pt1 = o3d.geometry.PointCloud()
    # pt1.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
    # pt1.colors = o3d.utility.Vector3dVector(np.zeros((100000,3)))
    # o3d.visualization.draw_geometries([pt1], width=50000, height=50000)
    # pt2 = o3d.geometry.PointCloud()
    # pt2.points = o3d.utility.Vector3dVector(gt.reshape(-1, 3))
    # o3d.visualization.draw_geometries([pt2], width=50000, height=50000)



    pc = np.expand_dims(pc, axis=0)
    gt = np.expand_dims(gt, axis=0)
    pred_tensor, centroid, furthest_distance = normalize_point_cloud(pc)
    gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt)
    cd_forward, cd_backward = nn_distance(pred_tensor, gt_tensor)
    cd_forward_value = cd_forward[0, :]
    cd_backward_value = cd_backward[0, :]
    md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
    hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
    cd_backward_value = np.mean(cd_backward_value)
    cd_forward_value = np.mean(cd_forward_value)
    CD = cd_forward_value+cd_backward_value
    hausdorff = hd_value
    print('CD:', float(CD), 'HD:', float(hausdorff))


# for D in [PRED_DIR]:
#     avg_md_forward_value = 0
#     avg_md_backward_value = 0
#     avg_hd_value = 0
#     avg_emd_value = 0
#     counter = 0
#     pred_paths = glob(os.path.join(D, "*.xyz"))
#
#     gt_pred_pairs = []
#     for p in pred_paths:
#         name, ext = os.path.splitext(os.path.basename(p))
#         assert(ext in (".ply", ".xyz"))
#         try:
#             gt = gt_paths[gt_names.index(name)]
#         except ValueError:
#             pass
#         else:
#             gt_pred_pairs.append((gt, p))
#
#     print("total inputs ", len(gt_pred_pairs))
#     tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
#     if tag:
#         tag = tag.groups()[0]
#     else:
#         tag = D
#
#     print("{:60s}".format(tag), end=' ')
#     global_p2f = []
#     global_density = []
#     global_uniform = []
#
#     with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
#         writer.writeheader()
#         for gt_path, pred_path in gt_pred_pairs:
#             row = {}
#             gt = load(gt_path)[:, :3]
#             gt = gt[np.newaxis, ...]
#             pred = pc_util.load(pred_path)
#             pred = pred[:, :3]
#
#             row["name"] = os.path.basename(pred_path)
#             pred = pred[np.newaxis, ...]
#             # cd_forward_value, cd_backward_value = sess.run([cd_forward, cd_backward], feed_dict={pred_placeholder:pred, gt_placeholder:gt})
#             pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred)
#             gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt)
#             cd_forward, cd_backward = nn_distance(pred_tensor, gt_tensor)
#             cd_forward_value = cd_forward[0, :]
#             cd_backward_value = cd_backward[0, :]
#             #save_ply_property(np.squeeze(pred), cd_forward_value, pred_path[:-4]+"_cdF.ply", property_max=0.003, cmap_name="jet")
#             #save_ply_property(np.squeeze(gt), cd_backward_value, pred_path[:-4]+"_cdB.ply", property_max=0.003, cmap_name="jet")
#             md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
#             hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
#             cd_backward_value = np.mean(cd_backward_value)
#             cd_forward_value = np.mean(cd_forward_value)
#             row["CD"] = cd_forward_value+cd_backward_value
#             row["hausdorff"] = hd_value
#             avg_md_forward_value += cd_forward_value
#             avg_md_backward_value += cd_backward_value
#             avg_hd_value += hd_value
#             if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.txt"):
#                 point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.txt")
#                 if point2mesh_distance.size == 0:
#                     continue
#                 point2mesh_distance = point2mesh_distance[:, 3]
#                 row["p2f avg"] = np.nanmean(point2mesh_distance)
#                 row["p2f std"] = np.nanstd(point2mesh_distance)
#                 global_p2f.append(point2mesh_distance)
#
#             if os.path.isfile(pred_path[:-4] + "_disk_idx.txt"):
#
#                 idx_file = pred_path[:-4] + "_disk_idx.txt"
#                 radius_file = pred_path[:-4] + '_radius.txt'
#                 map_points_file = pred_path[:-4] + '_point2mesh_distance.txt'
#
#                 disk_measure = analyze_uniform(idx_file, radius_file, map_points_file)
#                 global_uniform.append(disk_measure)
#
#                 for i in range(2):
#                     row["uniform_%d" % i] = disk_measure[i, 0]
#
#             writer.writerow(row)
#             counter += 1
#
#         row = OrderedDict()
#
#         avg_md_forward_value /= counter
#         avg_md_backward_value /= counter
#         avg_hd_value /= counter
#         avg_emd_value /= counter
#         avg_cd_value = avg_md_forward_value + avg_md_backward_value
#         row["CD"] = avg_cd_value
#         row["hausdorff"] = avg_hd_value
#         row["EMD"] = avg_emd_value
#         if global_p2f:
#             global_p2f = np.concatenate(global_p2f, axis=0)
#             mean_p2f = np.nanmean(global_p2f)
#             std_p2f = np.nanstd(global_p2f)
#             row["p2f avg"] = mean_p2f
#             row["p2f std"] = std_p2f
#
#         if global_uniform:
#             global_uniform = np.array(global_uniform)
#             uniform_mean = np.mean(global_uniform, axis=0)
#             for i in range(precentages.shape[0]):
#                 row["uniform_%d" % i] = uniform_mean[i, 0]
#
#         writer.writerow(row)
#         print("|".join(["{:>15.8f}".format(d) for d in row.values()]))
#
