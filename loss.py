import numpy as np
import math
import tensorflow as tf
import os
import sys
import op
import get_feature
from tf_ops.nn_distance import tf_nndistance
from tf_ops.approxmatch import tf_approxmatch
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point,knn_point_2
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
# from sampling.tf_sampling import gather_point, farthest_point_sample

def discriminator_loss(D, input_real, input_fake): #adversarial loss
    real = D(input_real)
    fake = D(input_fake)
    real_loss = tf.reduce_mean(tf.square(real-1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))
    # print(real_loss,fake_loss)
    loss = real_loss + fake_loss
    return loss, (1.0-real_loss), fake_loss

def generator_loss(D, input_fake):
    fake = D(input_fake)
    fake_loss = tf.reduce_mean(tf.square(fake-1.0))
    return fake_loss

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

def approx_match(xyz1,xyz2,b,n,m):
    match = np.zeros((b,m,n))

    for i in range(0,b):
        factorl = max(n, m) / n;
        factorr = max(n, m) / m;
        saturatedl = np.full(shape=n, fill_value=factorl, dtype=np.float32)
        saturatedr = np.full(shape=m, fill_value=factorr, dtype=np.float32)
        weight = np.zeros((m,n))
        j = 8
        while j >= -2:
            level= -math.pow(4, j)
            if (j == -2):
                level=0;
            dist = square_distance(tf.expand_dims(xyz1[i],axis=0),tf.expand_dims(xyz2[i],axis=0))
            weight=np.exp(level*tf.squeeze(dist,axis=0)) * saturatedr
            ss = np.full(shape=m, fill_value=1e-9, dtype=np.float32)
            s = np.full(shape=m, fill_value=1e-9, dtype=np.float32)
            s += tf.reduce_sum(weight,axis=-1)
            weight = weight / s * saturatedl
            # ss += weight
            ss += tf.reduce_sum(weight,axis=0)
            r = saturatedr / ss
            ss = tf.where(r<1, x=r, y=1)
            ss2 = np.full(shape=m, fill_value=0, dtype=np.float32)
            weight *=ss
            s = np.zeros((m))
            s = tf.reduce_sum(weight,axis=-1)
            ss2 = tf.reduce_sum(weight,axis=0)
            saturatedl = tf.where((saturatedl-s>0),x=(saturatedl-s),y=0)
            match[i] += weight
            saturatedr = tf.where((saturatedr-ss2>0),x=(saturatedr-ss2),y=0)
            j = j-1
    # match=tf.cast(match,dtype=tf.float32)
    return match

def match_cost(xyz1,xyz2,match,b,n,m):
    cost = None
    for i in range(0,b):
        dist = square_distance(tf.expand_dims(xyz1[i], axis=0), tf.expand_dims(xyz2[i], axis=0))
        dist = tf.squeeze(dist,axis=0)
        dist += 1e-10
        a = tf.sqrt(dist)
        # print('dist:',dist,'a:',a)
        d1 = a*match[i]
        d2 = tf.reduce_sum(d1,axis=-1)
        cost_=tf.reduce_sum(d2,axis=-1);
        cost_ = tf.reshape(cost_, [1])
        if cost is not None:
            cost = tf.concat([cost, cost_],axis=0)
        else:
            cost = cost_
    return cost

# def pc_distance(pcd1,pcd2, radius):
#     # assert pcd1.shape[1] == pcd2.shape[1]
#     B, N, _ = pcd1.shape
#     B, M, _ = pcd2.shape
#     dist = square_distance(pcd1, pcd2)
#     dist = tf.sort(dist, axis=-1)
#     dists = tf.slice(dist, begin=[0, 0, 0], size=[B, N, 1])
#     cost = tf.reduce_sum(dists, axis=1, keepdims=False)
#     cost = cost / radius
#     earth_mover = tf.reduce_mean(cost / N)
#     # d = wasserstein_distance(pcd1, pcd2)
#     return earth_mover

def pc_distance(pcd1,pcd2, radius):
    assert pcd1.shape[1] == pcd2.shape[1]
    B, N, _ = pcd1.shape
    B, M, _ = pcd2.shape
    match = approx_match(pcd1, pcd2, B, N, M)
    cost = match_cost(pcd1, pcd2, match, B, N, M)
    # cost = tf.reduce_sum(tf.norm(pcd2 - pcd1))
    cost = cost / radius
    cost = tf.reduce_mean(cost / N)
    return cost

def get_repulsion_loss(pred, nsample=20, radius=0.07, knn=False, use_l1=False, h=0.001):

    if knn:
        _, idx = knn_point_2(nsample, pred, pred)
        pts_cnt = tf.constant(nsample, shape=(30, 1024))
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    if use_l1:
        dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    else:
        dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    val, idx = tf.nn.top_k(-dists, 5)
    val = val[:, :, 1:]  # remove the first one

    if use_l1:
        h = np.sqrt(h)*2

    val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    repulsion_loss = tf.reduce_mean(val)
    return repulsion_loss

def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
    B,N,C = pcd.shape
    npoint = int(N * 0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample
        #print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

        #expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis+1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis,axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean*math.pow(p*100,2)
        #nothing 4
        loss.append(mean)
    return tf.add_n(loss)/len(percentages), loss

def cd_loss(xy1, xy2, radius):
    B, N, _ = xy1.shape
    d1 = square_distance(xy2, xy1)
    d2 = square_distance(xy1, xy2)

    idx1 = tf.argmin(d1, axis=-1)
    dist1 = tf.reduce_min(d1, axis=-1)

    idx2 = tf.argmin(d2, axis=-1)
    dist2 = tf.reduce_min(d2, axis=-1)

    # cost1 = tf.reduce_sum(dist1, axis=1, keepdims=False)
    # cost2 = tf.reduce_sum(dist2, axis=1, keepdims=False)
    cost1 = tf.reduce_mean(dist1, axis=1, keepdims=False)
    cost2 = tf.reduce_mean(dist2, axis=1, keepdims=False)
    CD_dist = 0.5 * dist1 + 0.5 * dist2
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist / radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss , cost1, cost2, dist1, dist2, idx1, idx2

def calc_dcd(x, gt, radius, alpha=50, n_lambda=0.0, non_reg=False):

    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    #_, cd_p, cd_t, dist1, dist2, idx1, idx2 = cd_loss(x, gt, radius)
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(x, gt)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    alpha = tf.expand_dims(tf.cast(alpha, dtype=tf.float32),axis=-1)
    exp_dist1 = tf.exp(-dist1 * tf.cast(alpha, dtype=tf.float32))
    exp_dist2 = tf.exp(-dist2 * tf.cast(alpha, dtype=tf.float32))

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = np.bincount(idx1[b])
        # weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = count1[idx1[b]]** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append(tf.reduce_mean(- exp_dist1[b] * weight1 + 1.))

        count2 = np.bincount(idx2[b])
        weight2 = count2[idx2[b]]** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append(tf.reduce_mean(- exp_dist2[b] * weight2 + 1.))

    loss1 = tf.stack(loss1)
    loss2 = tf.stack(loss2)
    loss = (loss1 + loss2) / 2

    loss = tf.reduce_mean(loss)

    return loss

# def to_uniform_dis(pcd, B, N, p, radius, npoint):
#     nsample = int(N * p)
#     r = math.sqrt(p * radius)
#     pc = np.array(pcd)
#     idx = op.farthest_point_sample(pc, npoint)
#     new_xyz = np.zeros((B, npoint, 3), dtype=np.float32)  # (batch_size, npoint, 3)
#     for i in range(B):
#         new_xyz[i, :, :] = pc[i, idx[i], :]  # (batch_size, npoint, 3)
#     idx = op.query_ball_point(r, nsample, pcd, new_xyz, npoint)
#     grouped = None
#     for i in range(B):
#         # grouped_pcd[i, :, :] = pc[i, idx[i], :]
#         grouped_pcd = tf.slice(pcd, begin=[i, 0, 0], size=[1, tf.shape(pcd)[1], tf.shape(pcd)[2]])
#         idx_ = tf.slice(idx, begin=[i, 0, 0], size=[1, tf.shape(idx)[1], tf.shape(idx)[2]])
#         grouped_pcd1 = op.index_points(grouped_pcd, idx_)
#         if grouped is not None:
#             grouped = tf.concat([grouped, grouped_pcd1], axis=0)
#         else:
#             grouped = grouped_pcd1
#     grouped_pcd = tf.concat(tf.unstack(grouped, axis=1), axis=0)
#
#     var, _ = get_feature.knn_point(2, grouped_pcd, grouped_pcd)
#     uniform_dis = -var[:, :, 1:]
#     uniform_dis = tf.sqrt(tf.abs(uniform_dis + 1e-8))
#     uniform_dis = tf.reduce_mean(uniform_dis, axis=[-1])
#     return uniform_dis
#
# def get_uniform_loss(pcd1, pcd2, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
#     B,N,C = pcd1.shape
#     npoint = int(N * 0.05)
#     loss=[]
#     for p in percentages:
#         uniform_dis = to_uniform_dis(pcd1, B, N, p, radius, npoint)
#         expect_len = to_uniform_dis(pcd2, B, N, p, radius, npoint)
#         uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
#         uniform_dis = tf.reshape(uniform_dis, [-1])
#
#         mean, variance = tf.nn.moments(uniform_dis, axes=0)
#         mean = mean*math.pow(p*100,2)
#         #nothing 4
#         loss.append(mean)
#     return tf.add_n(loss)/len(percentages)