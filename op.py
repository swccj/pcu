import tensorflow as tf
import numpy as np
import os
import sys
import get_feature
import up_down

# def mlp(features, layer_dims):
#     for i, num_outputs in enumerate(layer_dims[:-1]):
#         features = tf.keras.layers.Dense(units=num_outputs, activation='relu')(features)
#     outputs = tf.keras.layers.Dense(units=layer_dims[-1], activation=None)(features)
#     return outputs
#
#
# def mlp_conv(inputs, layer_dims):
#     for i, num_out_channel in enumerate(layer_dims[:-1]):
#         inputs = tf.keras.layers.Conv2D(num_out_channel, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal())(inputs)
#         # inputs = tf.compat.v1.layers.conv2d(inputs, filters=num_out_channel, kernel_size=1, strides=1, padding='valid',activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal())
#     outputs = tf.keras.layers.Conv2D(layer_dims[-1], kernel_size=1, strides=1, padding='same',activation=None, kernel_initializer=tf.keras.initializers.glorot_normal())(inputs)
#     # outputs = tf.compat.v1.layers.conv2d(inputs, filters=layer_dims[-1], kernel_size=1, strides=1, padding='valid',activation=None,kernel_initializer=tf.keras.initializers.glorot_normal())
#     # outputs = tf.cast(outputs, tf.float32)
#     return outputs
#
# def conv1d(inputs,filters,kernel_size,padding='SAME',activations=tf.nn.relu, use_xavier=True,bn=False,use_bias = True,weight_decay=0.00001, bn_decay=None,is_training = None):
#     if use_xavier:
#         initializer = tf.keras.initializers.glorot_normal()
#     else:
#         initializer = tf.truncated_normal_initializer(stddev=stddev)
#     output = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding=padding, use_bias=use_bias,
#                                     bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay),kernel_initializer=initializer,
#                                     kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(inputs)
#     # output = tf.compat.v1.layers.conv1d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding='valid',
#     #                                     kernel_initializer=initializer,kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
#     #                                     bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay), use_bias=use_bias)
#     # output = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding=padding, activation=None)(inputs)
#     if bn:
#         output = tf.keras.layers.BatchNormalization(axis=-1)(output, training = is_training)
#     # outputs = tf.compat.v1.layers.batch_normalization(output, training=is_training, renorm=False, fused=True)
#
#     if activations is not None:
#         output = activations(output)
#     # output = tf.cast(output, tf.float32)
#     return output
#
# def conv2d(inputs,filters,h,w,padding='SAME',activations=tf.nn.relu,use_xavier=True,bn=False,stddev=1e-3,use_bias = True,bn_decay=None,weight_decay=0.00001, is_training = None):
#     if use_xavier:
#         initializer = tf.keras.initializers.glorot_normal()
#     else:
#         initializer = tf.truncated_normal_initializer(stddev=stddev)
#     output = tf.keras.layers.Conv2D(filters, (h,w), strides=1, padding=padding, use_bias=use_bias,
#                                     bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay),kernel_initializer=initializer,
#                                     kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(inputs)
#     # output = tf.compat.v1.layers.conv2d(inputs, filters=filters, kernel_size=(h,w), strides=(1, 1), padding='valid',
#     #                            kernel_initializer=initializer,kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
#     #                            bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay),use_bias=use_bias)
#     # output = tf.keras.layers.Conv2D(filters, (h,w), strides=1, padding=padding, activation=None)(inputs)
#     if bn:
#         output = tf.keras.layers.BatchNormalization(axis=-1)(output, training = is_training)
#     # outputs = tf.compat.v1.layers.batch_normalization(inputs=output, training=is_training, renorm=False, fused=True)
#
#     if activations is not None:
#         output = activations(output)
#     # output = tf.cast(output, tf.float32)
#     return output
#
# def feature_extraction(inputs, is_training=True, bn_decay=None):
#     growth_rate = 24
#     dense_n = 3
#     knn = 16
#     comp = growth_rate*2
#     l0 = tf.expand_dims(inputs, axis=2)
#     l0 = conv2d(l0, 24, 1, 1, padding='valid', activations=None, is_training=is_training)
#     l0 = tf.squeeze(l0,axis=2)
#
#     l1 , l1_idx = get_feature.dense_conv(l0, n=dense_n, growth_rate=growth_rate, k=knn, is_training=is_training)
#     l1 = tf.concat([l1,l0], axis=-1)
#
#     l2 = conv1d(l1, comp, 1, padding='valid',is_training=is_training)
#     l2 , l2_idx = get_feature.dense_conv(l2, n=dense_n, growth_rate=growth_rate, k=knn, is_training=is_training)
#     l2 = tf.concat([l2,l1], axis=-1)
#
#     l3 = conv1d(l2, comp, 1, padding='valid',is_training=is_training)
#     l3 , l3_idx = get_feature.dense_conv(l3, n=dense_n, growth_rate=growth_rate, k=knn, is_training=is_training)
#     l3 = tf.concat([l3,l2], axis=-1)
#
#     l4 = conv1d(l3, comp, 1, padding='valid', is_training=is_training)
#     l4 , l4_idx = get_feature.dense_conv(l4, n=dense_n, growth_rate=growth_rate, k=knn, is_training=is_training)
#     l4 = tf.concat([l4,l3], axis=-1)
#     l4 = tf.expand_dims(l4, axis=2)
#     return l4
#
# def up_projection_unit(inputs, up_ratio, is_training=True, bn_decay=None):
#     L = conv2d(inputs, 128, 1, 1, padding='valid',is_training=is_training)
#     H0 = up_down.up_block(L, up_ratio, is_training=is_training, bn_decay=bn_decay)
#     L0 = up_down.down_block(H0, up_ratio, is_training=is_training, bn_decay=bn_decay)
#     E0 = L0-L
#     H1 = up_down.up_block(E0, up_ratio,is_training=is_training, bn_decay=bn_decay)
#     H2 = H0+H1
#     return H2

class mlp_dense(tf.keras.layers.Layer):
    def __init__(self,layer_dims,activations=tf.nn.relu):
        super(mlp_dense, self).__init__()
        self.layer_dims = layer_dims
        for i, num_outputs in enumerate(self.layer_dims[:-1]):
            self.dense1 = tf.keras.layers.Dense(units=num_outputs, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.layer_dims[-1])
        self.activation = activations

    def call(self, inputs):
        features1 = self.dense1(inputs)
        features2 = self.dense2(features1)
        if self.activation is not None:
            features2 = self.activation(features2)
        return features2

class mlp_conv2d(tf.keras.layers.Layer):
    def __init__(self,layer_dims,activations=tf.nn.relu):
        super(mlp_conv2d, self).__init__()
        self.layer_dims = layer_dims
        for i, num_out_channel in enumerate(self.layer_dims[:-1]):
            self.conv1 = tf.keras.layers.Conv2D(num_out_channel, kernel_size=1, strides=1, padding='same',  kernel_initializer=tf.keras.initializers.glorot_normal())
        self.conv2 = tf.keras.layers.Conv2D(self.layer_dims[-1], kernel_size=1, strides=1, padding='same', activation=None,kernel_initializer=tf.keras.initializers.glorot_normal())
        self.activation = activations

    def call(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.conv2(output1)
        if self.activation is not None:
            output2 = self.activation(output2)
        return output2

class conv1d(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,padding='SAME',activations=tf.nn.relu, bn=False,use_bias = True,weight_decay=0.00001, bn_decay=None):
        super(conv1d, self).__init__()
        self.initializer = tf.keras.initializers.glorot_normal()
        self.BatchNormalization = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding=padding, use_bias=use_bias,bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
                                    kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))
        self.bn = bn
        self.activation = activations

    def call(self, input, training=True):
        output = self.conv(input)
        if self.bn:
            output = self.BatchNormalization(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

class conv2d(tf.keras.layers.Layer):
    def __init__(self,filters,h,w,padding='valid',activations=tf.nn.relu,bn=False,stddev=1e-3,use_bias = True,bn_decay=None,weight_decay=0.00001):
        super(conv2d, self).__init__()
        self.initializer = tf.keras.initializers.glorot_normal()
        self.BatchNormalization = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv = tf.keras.layers.Conv2D(filters, (h, w), strides=1, padding=padding, use_bias=use_bias,bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
                                    kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))
        self.bn = bn
        self.activation = activations

    def call(self, input, training=True):
        output = self.conv(input)
        if self.bn:
            output = self.BatchNormalization(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

class transformer_head(tf.keras.layers.Layer):
    def __init__(self, in_channel, dim, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(transformer_head, self).__init__()
        self.dim = dim
        self.pos_hidden_dim = pos_hidden_dim
        self.attn_hidden_multiplier = attn_hidden_multiplier
        self.in_channel = in_channel

        self.pos_conv1 = conv2d(self.pos_hidden_dim, 1, 1, bn=True)
        self.pos_conv2 = conv2d(self.dim, 1, 1, activations=None)

        self.attn_mlp1 = conv2d(self.dim*self.attn_hidden_multiplier, 1, 1, bn=True)
        self.attn_mlp2 = conv2d(self.dim, 1, 1, activations=None)


        #self.linear_end = conv1d(self.in_channel, 1, activations=None)

    def call(self, key, query, value, pos, idx):
        key = tf.gather_nd(key, idx)
        qk_rel = tf.expand_dims(query, axis=2) - key
        pos_rel = tf.expand_dims(pos, axis=2) - tf.gather_nd(pos, idx)
        pos_em = self.pos_conv1(pos_rel)
        pos_em = self.pos_conv2(pos_em)

        attention = self.attn_mlp1(qk_rel + pos_em)
        attention = self.attn_mlp2(attention)
        dk = tf.cast(tf.shape(key)[-1],tf.float32)
        attention = attention / tf.sqrt(dk)
        attention = tf.nn.softmax(attention)

        value = tf.expand_dims(value, axis=2) + pos_em
        agg = tf.einsum('b c i j, b c i j -> b c j', attention, value)
       # y = self.linear_end(agg)
        return agg

class transformer_mu(tf.keras.layers.Layer):
    def __init__(self, in_channel, dim, knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(transformer_mu, self).__init__()
        self.n_knn = knn
        self.dim = dim
        self.pos_hidden_dim = pos_hidden_dim
        self.attn_hidden_multiplier = attn_hidden_multiplier
        self.in_channel = in_channel
        self.linear_start = conv1d(self.dim, 1, activations=None)
        self.conv_key = conv1d(self.dim, 1, activations=None)
        self.conv_query = conv1d(self.dim, 1, activations=None)
        self.conv_value = conv1d(self.dim, 1, activations=None)
        self.mu_head1 = transformer_head(self.in_channel//3, self.dim//3, pos_hidden_dim=self.pos_hidden_dim//3,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//3)
        self.mu_head2 = transformer_head(self.in_channel//3, self.dim//3, pos_hidden_dim=self.pos_hidden_dim//3,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//3)
        self.mu_head3 = transformer_head(self.in_channel//3, self.dim//3, pos_hidden_dim=self.pos_hidden_dim//3,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//3)

    def call(self, x, pos):
        identity = x
        x = self.linear_start(x)
        b, n, dim = x.shape
        _, idx = get_feature.knn_point_2(self.n_knn, pos, pos)
        key = self.conv_key(x)
        value = self.conv_query(x)
        query = self.conv_value(x)
        key1, key2, key3 = tf.split(value=key, num_or_size_splits=3, axis=-1)
        query1, query2, query3 = tf.split(value=query, num_or_size_splits=3, axis=-1)
        value1, value2, value3 = tf.split(value=value, num_or_size_splits=3, axis=-1)

        y1 = self.mu_head1(key1, query1, value1, pos, idx)
        y2 = self.mu_head2(key2, query2, value2, pos, idx)
        y3 = self.mu_head3(key3, query3, value3, pos, idx)
        y = tf.concat((y1, y2, y3), axis=-1)
        return y + identity

class transformer(tf.keras.layers.Layer):
    def __init__(self, in_channel, dim, knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(transformer, self).__init__()
        self.n_knn = knn
        self.dim = dim
        self.pos_hidden_dim = pos_hidden_dim
        self.attn_hidden_multiplier = attn_hidden_multiplier
        self.in_channel = in_channel
        self.conv_key = conv1d(self.dim, 1, activations=None)
        self.conv_query = conv1d(self.dim, 1, activations=None)
        self.conv_value = conv1d(self.dim, 1, activations=None)

        self.pos_conv1 = conv2d(self.pos_hidden_dim, 1, 1, bn=True)
        self.pos_conv2 = conv2d(self.dim, 1, 1, activations=None)

        self.attn_mlp1 = conv2d(self.dim*self.attn_hidden_multiplier, 1, 1, bn=True)
        self.attn_mlp2 = conv2d(self.dim, 1, 1, activations=tf.nn.softmax)

        self.linear_start = conv1d(self.dim, 1, activations=None)
        self.linear_end = conv1d(self.in_channel, 1, activations=None)

    def call(self, x, pos):
        identity = x
        x = self.linear_start(x)
        b, n, dim = x.shape
        _, idx = get_feature.knn_point_2(self.n_knn, pos, pos)
        key = self.conv_key(x)
        value = self.conv_query(x)
        query = self.conv_value(x)

        key = tf.gather_nd(key, idx)
        qk_rel = tf.expand_dims(query, axis=2) - key
        pos_rel = tf.expand_dims(pos, axis=2) - tf.gather_nd(pos, idx)
        pos_em = self.pos_conv1(pos_rel)
        pos_em = self.pos_conv2(pos_em)

        attention = self.attn_mlp1(qk_rel + pos_em)
        attention = self.attn_mlp2(attention)

        value = tf.expand_dims(value, axis=2) + pos_em
        agg = tf.einsum('b c i j, b c i j -> b c j', attention, value)
        y = self.linear_end(agg)
        return y + identity

class transformer_cr(tf.keras.layers.Layer):
    def __init__(self, in_channel, dim, knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(transformer_cr, self).__init__()
        self.n_knn = knn
        self.dim = dim
        self.pos_hidden_dim = pos_hidden_dim
        self.attn_hidden_multiplier = attn_hidden_multiplier
        self.in_channel = in_channel
        self.linear_start1 = conv1d(self.dim, 1, activations=None)
        self.linear_start2 = conv1d(self.dim, 1, activations=None)
        self.conv_key = conv1d(self.dim, 1, activations=None)
        self.conv_query = conv1d(self.dim, 1, activations=None)
        self.conv_value = conv1d(self.dim, 1, activations=None)
        self.mu_head1 = transformer_head(self.in_channel//4, self.dim//4, pos_hidden_dim=self.pos_hidden_dim//4,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//3)
        self.mu_head2 = transformer_head(self.in_channel//4, self.dim//4, pos_hidden_dim=self.pos_hidden_dim//4,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//4)
        self.mu_head3 = transformer_head(self.in_channel//4, self.dim//4, pos_hidden_dim=self.pos_hidden_dim//4,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier//4)
        self.mu_head4 = transformer_head(self.in_channel // 4, self.dim // 4, pos_hidden_dim=self.pos_hidden_dim // 4,
                                         attn_hidden_multiplier=self.attn_hidden_multiplier // 4)
        self.linear_end = conv1d(self.in_channel, 1, activations=None)

    def call(self, xl, xs, pos):
        xl = self.linear_start1(xl)
        xs = self.linear_start2(xs)
        _, idx = get_feature.knn_point_2(self.n_knn, pos, pos)
        key = self.conv_key(xs)
        value = self.conv_query(xs)
        query = self.conv_value(xl)
        key1, key2, key3 ,key4 = tf.split(value=key, num_or_size_splits=4, axis=-1)
        query1, query2, query3, query4 = tf.split(value=query, num_or_size_splits=4, axis=-1)
        value1, value2, value3, value4 = tf.split(value=value, num_or_size_splits=4, axis=-1)

        y1 = self.mu_head1(key1, query1, value1, pos, idx)
        y2 = self.mu_head2(key2, query2, value2, pos, idx)
        y3 = self.mu_head3(key3, query3, value3, pos, idx)
        y4 = self.mu_head4(key4, query4, value4, pos, idx)
        y = tf.concat((y1, y2, y3, y4), axis=-1)
        y = self.linear_end(y+xl)
        return y 

class feature_extraction(tf.keras.layers.Layer):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.growth_rate = 24
        self.dense_n = 3
        self.knn = 16
        self.comp = self.growth_rate * 2
        self.dense1 = get_feature.dense_conv(n=self.dense_n, growth_rate=self.growth_rate, k=self.knn)
        self.dense2 = get_feature.dense_conv(n=self.dense_n, growth_rate=self.growth_rate, k=self.knn)
        self.dense3 = get_feature.dense_conv(n=self.dense_n, growth_rate=self.growth_rate, k=self.knn)
        self.dense4 = get_feature.dense_conv(n=self.dense_n, growth_rate=self.growth_rate, k=self.knn)
        self.conv2d = conv2d(24, 1, 1, padding='valid', activations=None)
        self.conv1d_1 = conv1d(self.comp, 1, padding='valid')
        self.conv1d_2 = conv1d(self.comp, 1, padding='valid')
        self.conv1d_3 = conv1d(self.comp, 1, padding='valid')
        self.transformer1 = transformer_cr(96, dim=64)
        self.transformer2 = transformer_cr(216, dim=64)
        self.transformer3 = transformer_cr(336, dim=128)
        self.transformer4 = transformer_cr(456, dim=128)

    def call(self, inputs):
        l0 = tf.expand_dims(inputs, axis=2)
        l0 = self.conv2d(l0)
        l0 = tf.squeeze(l0, axis=2)

        l1, l1_idx = self.dense1(l0)
        l1 = self.transformer1(l0, l1, inputs)
        l1 = tf.concat([l1, l0], axis=-1)
        

        l2 = self.conv1d_1(l1)
        l2, l2_idx = self.dense2(l2)
        l2 = self.transformer2(l1, l2, inputs)
        l2 = tf.concat([l2, l1], axis=-1)

        l3 = self.conv1d_2(l2)
        l3, l3_idx = self.dense3(l3)
        l3 = self.transformer3(l2, l3, inputs)
        l3 = tf.concat([l3, l2], axis=-1)

        l4 = self.conv1d_3(l3)
        l4, l4_idx = self.dense4(l4)
        l4 = self.transformer4(l3, l4, inputs)
        l4 = tf.concat([l4, l3], axis=-1)
        l4 = tf.expand_dims(l4, axis=2)
        return l4

class gru(tf.keras.Model):
    def __init__(self, c):
        super(gru, self).__init__()
        self.c = c
        self.convz = conv2d(self.c, 1, 1, bn=True, activations=tf.nn.sigmoid)
        self.convr = conv2d(self.c, 1, 1, bn=True, activations=tf.nn.sigmoid)
        self.convh = conv2d(self.c, 1, 1, bn=True, activations=tf.nn.relu)
    def call(self, h, x):
        # b, n, c = h.shape
        # h = tf.expand_dims(h, axis=2)
        # x = tf.expand_dims(x, axis=2)

        z = self.convz(tf.concat((h,x), axis=-1))
        r = self.convr(tf.concat((h,x), axis=-1))
        h_ = self.convh(tf.concat((h,r*x), axis=-1))
        h = (1 - z) * h + z * h_
        # h = tf.squeeze(h, [2])
        return h



class up_projection_unit(tf.keras.layers.Layer):
    def __init__(self, up_ratio):
        super(up_projection_unit, self).__init__()
        self.up_ratio = up_ratio
        self.conv2d1 = conv2d(128, 1, 1, padding='valid')
        # self.conv2d2 = conv2d(128, 1, 1, padding='valid')
        self.up_1 = up_down.up_block(is_training=True, up_ratio=self.up_ratio)
        self.up_2 = up_down.up_block(is_training=True, up_ratio=self.up_ratio)
        # self.up_3 = up_down.up_block(is_training=True, up_ratio=self.up_ratio)
        self.down1 = up_down.down_block(self.up_ratio)
        # self.down2 = up_down.down_block(self.up_ratio)
        #self.gru1 = gru(128)
        self.gru2 = gru(128)

    def call(self, features):
        L = self.conv2d1(features)
        H0 = self.up_1(L)
        L0 = self.down1(H0)
        E0 = L0 - L
        # e = self.conv2d2(tf.expand_dims(xyz, axis=2))
        #E0 = self.gru1(L0,L)
        H1 = self.up_2(E0)
        # H2 = H0 + H1
        H2 = self.gru2(H0,H1)
        # L1 = self.down2(H2)
        # L1 = self.gru1(L1, L0)
        # E1 = L1 - L
        # H3 = self.up_3(E1, self.up_ratio)
        # H3 = self.gru2(H3, H1)
        # H4 = H0 + H3
        return H2

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

def query_ball_point(radius, nsample, xyz, new_xyz,npoint):#寻找球形领域中的点
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = np.arange(N, dtype=np.int32)
    group_idx = np.resize(group_idx,[1, 1, N])
    group_idx = tf.tile(group_idx,[B, S, 1])
    # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    sqrdists = tf.where(sqrdists < (radius ** 2), x=0, y=npoint)
    group_idx = tf.add(group_idx,sqrdists)
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = tf.sort(group_idx,axis=-1)
    # group_idx = tf.gather(group_idx,indices=nsample,axis=-1)
    group_idx = tf.slice(group_idx,begin=[0,0,0],size=[B,npoint,nsample])
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = tf.slice(group_idx,begin=[0,0,0],size=[B,npoint,1])
    # 找到group_idx中值等于N的点
    # 将这些点的值替换为第一个点的值
    group_idx = tf.where(group_idx >= N, x=group_first, y=group_idx)
    # group_idx[mask] = group_first[mask]
    return group_idx

def fps(xyz, npoint):#最远的采样
    B, N, C = xyz.shape
    centroids = []#存储索引位置
    distance = np.ones((B, N),dtype=np.float32) * 1e10#点间距离
    farthest = np.random.randint(0, N, (B,) ,dtype=np.int64)#当前最远点
    # batch_indices = np.arange(B, dtype=np.int64)
    for i in range(npoint):
    	# 更新第i个最远点
        centroids.append(farthest)
        # 取出这个最远点的xyz坐标
        # centroid = xyz[batch_indices, farthest, :]
        centroid = tf.gather(xyz,indices=farthest,axis=1)
        # centroid = tf.reshape(centroid,[B,1,3])
        # centroid = tf.slice(xyz,[batch_indices,farthest,0],[1,1,3])
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = tf.reduce_sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        # mask = dist < distance
        # distance[mask] = dist[mask]
        distance = tf.minimum(distance, dist)
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = tf.argmax(distance,axis=1,output_type=np.int64)
        # farthest = tf.reduce_max(farthest,keepdims=True)
    # centroids = np.array(centroids)
    # centroids = np.resize(centroids, [1,1024])
    # centroids = tf.expand_dims(centroids , 0)
    centroids = tf.transpose(centroids,[1,0])
    return centroids

def fps_(input,batch_size,out_num_point):
    # output = np.array(input)
    idx = farthest_point_sample(input, out_num_point)
    outputs = np.zeros((batch_size, out_num_point, 3), dtype=np.float32)
    for i in range(batch_size):
        outputs[i, :, :] = output[i, idx[i], :]
    return outputs

class ppd(tf.keras.layers.Layer):
    def __init__(self, m,up):
        super(ppd, self).__init__()
        self.m=m
        self.up = up
        self.m1 = self.m*2
        self.m2 = self.m*(self.up//2)

        self.conv1 = conv2d(self.m1, 1, 1, activations=None)
        # self.conv2 = conv2d(self.m2, 1, 1, activations=None)
        self.conv3 = conv2d(self.m1*3, 1, 1, activations=None)

        # self.gru1 = gru(3)
        # self.gru2 = gru(3)

    def call(self, point):
        # point = tf.expand_dims(point, axis=2)
        b, n ,_, c = point.shape
        pc1_feat = self.conv1(point)#256,256
        pc1 = tf.reshape(pc1_feat, (b, n, 2, self.m))#256,2,128

        pc1 = pc1 + point
        pc1 = tf.reshape(pc1, (b, -1, 1, self.m))

        # pc2_feat = self.conv2(pc1)
        pc2_feat = self.conv3(pc1_feat)
        pc2 = tf.reshape(pc2_feat, (b, -1, 3, self.m))

        pc2 = pc2 + pc1
        pc2 = tf.reshape(pc2, (b, -1, 1, self.m))

        return pc2