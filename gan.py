import keras
import tensorflow as tf
import op
import up_down
import numpy as np
import sys
import pc_util
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
# sys.setrecursionlimit(1000000)

class Generator(tf.keras.Model):
    def __init__(self, opts, is_training):
        super(Generator, self).__init__()
        self.opts = opts
        self.is_training = is_training
        self.num_point = self.opts.num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point*self.up_ratio)
        self.feature_extraction = op.feature_extraction()
        self.up = op.up_projection_unit(self.up_ratio_real)
        #self.up = op.up_projection_unit(self.up_ratio)
        self.conv1 = op.conv2d(64, 1, 1)
        self.conv2 = op.conv2d(3, 1, 1,activations=None, weight_decay=0.0)
        # self.transformer = op.transformer_mu(480, dim=120)
        # self.ppd = op.ppd(crop_point_num=1024, c=1024, m=1024, m1=64, m2=256)

    def call(self, inputs):
        features = self.feature_extraction(inputs)
        # features = tf.reduce_max(features, axis=1)
        # xyz_256, xyz_1024 = self.ppd(features)
        # H = self.transformer(tf.squeeze(features,axis=2),inputs)
        # H = self.up(tf.expand_dims(H,axis=2))
        H = self.up(features)
        coord = self.conv1(H)
        coord = self.conv2(coord)
        outputs = tf.squeeze(coord, [2])
        outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
        #fp_id = pc_util.FarthestPointSampling_ForBatch(outputs, self.out_num_point)
        #output = pc_util.grouping_operation(outputs, fp_id)
        # output_ = None
        # for i in range(self.opts.batch_size):
        #     output = tf.slice(outputs, begin=[i, 0, 0], size=[1, tf.shape(outputs)[1], tf.shape(outputs)[2]])
        #     output_f = op.fps(output, self.out_num_point)
        #     output_f = op.index_points(output, output_f)
        #     if output_ is not None:
        #         output_ = tf.concat([output_, output_f], axis=0)
        #     else:
        #         output_ = output_f
        return outputs

class Discriminator(tf.keras.Model):
    def __init__(self,  is_training):
        super(Discriminator, self).__init__()
        # self.opts = opts
        self.is_training = is_training
        self.bn = False
        self.start_number = 32
        self.mlp1 = op.mlp_conv2d([self.start_number, self.start_number * 2], activations=None)
        self.mlp2 = op.mlp_conv2d([self.start_number * 4, self.start_number * 8], activations=None)
        self.mlp3 = op.mlp_dense([self.start_number * 8, 1], activations=None)
        self.attention = up_down.attention_unit_d(is_training=self.is_training)

    def __call__(self, inputs):
        input = tf.expand_dims(inputs, axis=2)
        features0 = self.mlp1(input)
        features_global = tf.reduce_max(features0, axis=1, keepdims=True)
        features1 = tf.concat([features0, tf.tile(features_global, [1, tf.shape(inputs)[1], 1, 1])], axis=-1)
        features2 = self.attention(features1)
        features3 = self.mlp2(features2)
        features4 = tf.reduce_max(features3, axis=1)
        output = self.mlp3(features4)
        outputs = tf.reshape(output, [-1, 1])

        return outputs

def generator(batch_size,num_point):
    up_ratio = 4
    up_ratio_real = 4
    out_num_point = int(num_point * up_ratio)
    is_training = True
    input = tf.keras.layers.Input(shape=[num_point, 3])
    inputs = tf.reshape(input, [batch_size, num_point, 3])
    features = op.feature_extraction(inputs, is_training=is_training, bn_decay=None)
    H = op.up_projection_unit(features, up_ratio_real, is_training=is_training, bn_decay=None)
    coord = op.conv2d(H, 64, 1, 1, padding='valid', is_training=is_training)
    coord = op.conv2d(coord, 3, 1, 1, padding='valid', activations=None, weight_decay=0.0, is_training=is_training)
    outputs = tf.squeeze(coord, [2])
    ########################
    # output = op.farthest_point_sample_(outputs, out_num_point)
    # output_ = None
    # for i in range(batch_size):
    #     output = tf.slice(outputs, begin=[i, 0, 0], size=[1, tf.shape(outputs)[1], tf.shape(outputs)[2]])
    #     output_f = op.fps(output, out_num_point)
    #     output_f = op.index_points(output, output_f)
    #     if output_ is not None:
    #         output_ = tf.concat([output_, output_f], axis=0)
    #     else:
    #         output_ = output_f
    # idx = tf.numpy_function(op.farthest_point_sample, [outputs, out_num_point], tf.int64)
    # idx = tf.reshape(idx,[batch_size, out_num_point])
    # output = None
    # for i in range(batch_size):
    #     # grouped_pcd[i, :, :] = pc[i, idx[i], :]
    #     out = tf.slice(outputs, begin=[i, 0, 0], size=[1, tf.shape(outputs)[1], tf.shape(outputs)[2]])
    #     idx_ = tf.slice(idx, begin=[i, 0], size=[1, out_num_point])
    #     out_ = op.index_points(out, idx_)
    #     if output is not None:
    #         output = tf.concat([output, out_], axis=0)
    #     else:
    #         output = out_
    ########################
    model = tf.keras.Model(inputs=input, outputs=outputs)
    return model

def discriminator(batch_size,patch_num_point):
    start_number = 32
    is_training = True
    input = tf.keras.layers.Input(shape=[patch_num_point, 3])
    input_ = tf.reshape(input, [batch_size, patch_num_point, 3])
    inputs = tf.expand_dims(input_, axis=2)
    features0 = op.mlp_conv(inputs, [start_number, start_number * 2])
    features_global = tf.reduce_max(features0, axis=1, keepdims=True, name='maxpool_0')
    features1 = tf.concat([features0, tf.tile(features_global, [1, tf.shape(inputs)[1], 1, 1])],
                          axis=-1)
    features2 = up_down.attention_unit(features1, is_training=is_training)
    features3 = op.mlp_conv(features2, [start_number * 4, start_number * 8])
    features4 = tf.reduce_max(features3, axis=1, name='maxpool_1')
    output = op.mlp(features4, [start_number * 8, 1])
    outputs = tf.reshape(output, [-1, 1])
    model = tf.keras.Model(inputs=input, outputs=outputs)

    return model