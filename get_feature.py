import tensorflow as tf
import numpy as np
import os
import sys
import op

def square_distance(a, b):#确定每一个点距离采样点的距离
    # A shape is (N, P_A, C), B shape is (N, P_B, C)
    # D shape is (N, P_A, P_B)
    B, N, _ = a.shape
    _, M, _ = b.shape
    dist = -2 * tf.matmul(a, tf.transpose(b,(0,2,1))) # 2*(xn * xm + yn * ym + zn * zm)
    a_ = tf.reduce_sum(a**2 , axis=-1,keepdims=True)
    dist = tf.add(dist , tf.reshape(a_,[B,N,1]))
    b_ = tf.reduce_sum(b**2 , axis=-1,keepdims=True)
    dist = tf.add(dist , tf.reshape(b_,[B,1,M]))
    return dist

def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)#去重并排序
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated

def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_function(find_duplicate_columns, [A], tf.int32)
    # a = find_duplicate_columns(A)
    # D = tf.cast(D, tf.float32)
    D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    # b = xyz1.get_shape()[0].value
    # n = xyz1.get_shape()[1].value
    # c = xyz1.get_shape()[2].value
    # m = xyz2.get_shape()[1].value
    # xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    # xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    xyz1 = tf.expand_dims(xyz1,axis=1)
    xyz2 = tf.expand_dims(xyz2,axis=2)
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    #dist = tf.sqrt(tf.abs(dist))
    # outi, out = select_top_k(k, dist)
    # idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    # val = tf.slice(out, [0,0,0], [-1,-1,k])

    val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

def knn_point_2(k, points, queries, sort=True, unique=True):
    """
    points: dataset points (N, P0, K)
    queries: query points (N, P, K)
    return indices is (N, P, K, 2) used for tf.gather_nd(points, indices)
    distances (N, P, K)
    """
    batch_size = tf.shape(queries)[0]
    point_num = tf.shape(queries)[1]
    D = square_distance(queries, points)
    if unique:
        prepare_for_unique_top_k(D, points)
    distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    return -distances, indices

def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

# def dense_conv(feature, n=3,growth_rate=64, k=16, is_training = None):
#     y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
#     for i in range(n):
#         if i == 0:
#             y_1 = op.conv2d(y,growth_rate,1,1, padding='valid',is_training=is_training)#,bn=True)
#             y_2 = tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])
#             y = tf.concat([y_1,y_2], axis=-1)
#         elif i == n - 1:
#             y_1 = op.conv2d(y,growth_rate,1,1, padding='valid',activations=None, bn=False,is_training=is_training)
#             y = tf.concat([y_1,y], axis=-1)
#         else:
#             y_1 = op.conv2d(y,growth_rate,1,1, padding='valid',is_training=is_training)#,bn=True)
#             y = tf.concat([y_1,y], axis=-1)
#     y = tf.reduce_max(y, axis=-2)
#     return y, idx

class dense_conv(tf.keras.layers.Layer):
    def __init__(self, n, growth_rate, k):
        super(dense_conv, self).__init__()
        self.n = n
        self.growth_rate = growth_rate
        self.k = k
        self.conv2d_1 = op.conv2d(growth_rate, 1, 1, padding='valid')
        self.conv2d_2 = op.conv2d(growth_rate, 1, 1, padding='valid', activations=None)
        self.conv2d_3 = op.conv2d(growth_rate, 1, 1, padding='valid')

    def call(self, inputs):
        y, idx = get_edge_feature(inputs, k=self.k, idx=None)  # [B N K 2*C]
        for i in range(self.n):
            if i == 0:
                y_1 = self.conv2d_1(y)
                y_2 = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, self.k, 1])
                y = tf.concat([y_1,y_2], axis=-1)
            elif i == self.n - 1:
                y_1 = self.conv2d_2(y)
                y = tf.concat([y_1,y], axis=-1)
            else:
                y_1 = self.conv2d_3(y)
                y = tf.concat([y_1,y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx