import tensorflow as tf
import numpy as np
import os
import sys
import math
import op

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def gen_grid(up_ratio):
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():#__reversed__倒序迭代
        if (up_ratio%i)==0:
            num_x=i
            num_y=up_ratio//i
            break
    grid_x = tf.linspace(-0.2, 0.2, num_x)#tf.linspace(start, end, num)start到end间生成num个等间隔数
    grid_y = tf.linspace(-0.2, 0.2, num_y)

    x,y = tf.meshgrid(grid_x,grid_y)
    grid = tf.reshape(tf.stack([x,y],axis=1),[-1,2])#stack改变维度的拼接
    return grid

# def attention_unit(inputs, is_training=True):
#     dim = inputs.get_shape()[-1]#.value
#     layer = dim//4
#     f = op.conv2d(inputs, layer, 1, 1, padding='valid',is_training=is_training)
#     g = op.conv2d(inputs, layer, 1, 1, padding='valid', is_training=is_training)
#     h = op.conv2d(inputs, dim, 1, 1, padding='valid', is_training=is_training)
#
#     s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)#[bs,N,N]
#     beta = tf.nn.softmax(s, axis=-1)
#     o = tf.matmul(beta, hw_flatten(h))
#
#     # gamma = tf.compat.v1.get_variable("gamma", [1], initializer = tf.constant_initializer(0.0))
#     gamma = tf.Variable(tf.constant(0.0, shape=[1], name='gamma'))
#     o = tf.reshape(o, shape=inputs.shape)
#     x = gamma*o + inputs
#     return x
#
# def up_block(inputs, up_ratio, is_training=True, bn_decay=None):
#     net = inputs
#     dim = inputs.get_shape()[-1]
#     out_dim = dim*up_ratio
#     grid = gen_grid(up_ratio)
#     grid = tf.tile(tf.expand_dims(grid,0),[tf.shape(net)[0],1,tf.shape(net)[1]])#(b,n*4,2])
#     grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])#grid = tf.expand_dims(grid, axis=2)
#
#     net = tf.tile(net, [1, up_ratio, 1, 1])
#     net = tf.concat([net, grid], axis=-1)
#
#     net = attention_unit(net, is_training=is_training)
#
#     net = op.conv2d(net, 256, 1, 1, padding='valid', is_training=is_training)
#     net = op.conv2d(net, 128, 1, 1, padding='valid',is_training=is_training)
#     return net
#
# def down_block(inputs, up_ratio, is_training=True, bn_decay=None):
#     net = inputs
#     net1 = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
#     net2 = tf.transpose(net1, [0,2,1,3])
#
#     net3 = op.conv2d(net2, 256, 1, up_ratio, padding='valid',is_training=is_training)
#     net4 = op.conv2d(net3, 128, 1, 1, padding='valid', is_training=is_training)
#     return net4

class attention_unit(tf.keras.layers.Layer):
    def __init__(self, is_training):
        super(attention_unit, self).__init__()
        self.dim = 130
        self.layer = self.dim//4
        self.is_training = is_training
        self.conv1 = op.conv2d(self.layer, 1, 1, padding='valid')
        self.conv2 = op.conv2d(self.layer, 1, 1, padding='valid')
        self.conv3 = op.conv2d(self.dim, 1, 1, padding='valid')

    def hw_flatten(x):
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def call(self, inputs):
        # self.dim = inputs.get_shape()[-1]
        # self.layer = (self.dim)//4
        f = self.conv1(inputs)
        g = self.conv2(inputs)
        h = self.conv3(inputs)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs,N,N]
        beta = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(beta, hw_flatten(h))
        gamma = tf.Variable(tf.constant(0.0, shape=[1], name='gamma'))
        o = tf.reshape(o, shape=inputs.shape)
        x = gamma * o + inputs
        return x

class attention_unit_d(tf.keras.layers.Layer):
    def __init__(self, is_training):
        super(attention_unit_d, self).__init__()
        self.dim = 128
        self.layer = 32
        self.is_training = is_training
        self.conv1 = op.conv2d(self.layer, 1, 1, padding='valid')
        self.conv2 = op.conv2d(self.layer, 1, 1, padding='valid')
        self.conv3 = op.conv2d(self.dim, 1, 1, padding='valid')

    def hw_flatten(x):
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def call(self, inputs):
        # self.dim = inputs.get_shape()[-1]
        # self.layer = (self.dim)//4
        f = self.conv1(inputs)
        g = self.conv2(inputs)
        h = self.conv3(inputs)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs,N,N]
        beta = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(beta, hw_flatten(h))
        gamma = tf.Variable(tf.constant(0.0, shape=[1], name='gamma'))
        o = tf.reshape(o, shape=inputs.shape)
        x = gamma * o + inputs
        return x

class up_block(tf.keras.layers.Layer):
    def __init__(self, is_training, up_ratio):
        super(up_block, self).__init__()
        self.is_training = is_training
        self.attention = attention_unit(is_training=self.is_training)
        self.conv2d_1 = op.conv2d(256, 1, 1, padding='valid')
        self.conv2d_2 = op.conv2d(128, 1, 1, padding='valid')
        self.up_ratio = up_ratio
        self.ppd = op.ppd(m=128, up=self.up_ratio)

    def call(self, inputs):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim * self.up_ratio
        grid = gen_grid(self.up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1, tf.shape(net)[1]])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        # net = tf.tile(net, [1, self.up_ratio, 1, 1])
        net = self.ppd(net)
        net = tf.concat([net, grid], axis=-1)
        net = self.attention(net)
        net = self.conv2d_1(net)
        net = self.conv2d_2(net)
        return net

class down_block(tf.keras.layers.Layer):
    def __init__(self, up_ratio):
        super(down_block, self).__init__()
        self.up_ratio = up_ratio
        self.conv2d_1 = op.conv2d(256, 1, self.up_ratio, padding='valid')
        self.conv2d_2 = op.conv2d(128, 1, 1, padding='valid')

    def call(self, inputs):
        net = inputs
        net1 = tf.reshape(net, [tf.shape(net)[0], self.up_ratio, -1, tf.shape(net)[-1]])
        net2 = tf.transpose(net1, [0, 2, 1, 3])
        net3 = self.conv2d_1(net2)
        net4 = self.conv2d_2(net3)
        return net4