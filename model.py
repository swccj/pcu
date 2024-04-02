import tensorflow as tf
from gan import Generator,generator
from gan import Discriminator,discriminator
from vis import point_cloud_three_views
from data import Fetcher
import pc_util
from loss import pc_distance,get_uniform_loss,get_repulsion_loss,discriminator_loss,generator_loss,calc_dcd,cd_loss
import logging
import os
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
import op


class Model(object):
  def __init__(self,opts):
      self.opts = opts
      self.out_num_point = 1024

  def G_loss(self, G, D,input_x, input_y, radius,batch_idx,i):
      G_y = G(input_x)

      # x1_ = np.reshape(G_y, [-1, 3])
      # pathx1 = os.path.join(self.opts.out_folder[:], str(batch_idx) + str(i)+'z.xyz')
      # np.savetxt(pathx1, x1_, fmt='%.6f')
      # x = np.reshape(input_x, [-1, 3])
      # y = np.reshape(input_y, [-1, 3])
      # pathx = os.path.join(self.opts.out_folder[:], str(batch_idx) + str(i) + 'x.xyz')
      # pathy = os.path.join(self.opts.out_folder[:], str(batch_idx) + str(i) + 'y.xyz')
      # np.savetxt(pathx, x, fmt='%.6f')
      # np.savetxt(pathy, y, fmt='%.6f')

      # dis_loss =  self.opts.fidelity_w * pc_distance(G_y, input_y, radius)
      dcd = 100 * calc_dcd(G_y, input_y, radius)
      if self.opts.use_repulse:
          repulsion_loss = self.opts.repulsion_w * get_repulsion_loss(G_y) #tf.norm(get_repulsion_loss(G_y)-get_repulsion_loss(input_y))
      else:
          repulsion_loss = 0
      uniform_loss, loss =  get_uniform_loss(G_y)
      pu_loss =  dcd +  self.opts.uniform_w *uniform_loss + repulsion_loss + tf.compat.v1.losses.get_regularization_loss()
      G_gan_loss = self.opts.gan_w * generator_loss(D, G_y)
      G_loss = G_gan_loss + pu_loss
      # print(dis_loss,repulsion_loss,uniform_loss,G_gan_loss)
      return G_loss #, np.array(loss), dis_loss

  def D_loss(self,G, D, input_x,input_y, patch, radius):
      G_y = G(input_x)
      # cd, cost1, cost2, _,_,_,_ = cd_loss(G_y, input_y, radius)
      # hd_value = np.max(np.amax(cost1, axis=0) + np.amax(cost2, axis=0))
      # if patch % 50 == 0:
      #     D_loss = discriminator_loss(D, G_y, input_y)
      # else:
      #     D_loss = discriminator_loss(D, input_y, G_y)
      D_loss, real_loss, fake_loss = discriminator_loss(D, input_y, G_y)
      return D_loss, real_loss, fake_loss#, cd, hd_value

  def train(self):
      learning_rate_d = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=self.opts.base_lr_d, decay_steps=self.opts.lr_decay_steps_d,
          decay_rate=self.opts.lr_decay_rate)
      # learning_rate_d = self.opts.base_lr_d
      # learning_rate_d = tf.maximum(learning_rate_d, self.opts.lr_clip)
      learning_rate_g = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=self.opts.base_lr_g, decay_steps=self.opts.lr_decay_steps_g,
          decay_rate=self.opts.lr_decay_rate)
      # learning_rate_g = self.opts.base_lr_g
      # learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
      # self.G = Generator(self.opts,is_training=True)
      # self.G.build(input_shape=(self.opts.batch_size, self.opts.num_point,3))
      # D = Discriminator(self.opts,is_training=True)
      # self.D.build(input_shape=(self.opts.batch_size,self.opts.patch_num_point, 3))
      # self.g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_g, beta_1=self.opts.beta)
      # self.d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_d, beta_1=self.opts.beta)
      fetchworker = Fetcher(self.opts)
      fetchworker.start()
      # g_checkpoint = tf.train.Checkpoint(self.G, optimizer=self.g_optimizer)
      # d_checkpoint = tf.train.Checkpoint(self.D, optimizer=self.d_optimizer)
      # # ckpt管理器
      # ckpt_manager_g = tf.train.CheckpointManager(g_checkpoint, self.opts.save_g, max_to_keep=1)
      # ckpt_manager_d = tf.train.CheckpointManager(d_checkpoint, self.opts.save_d, max_to_keep=1)

      # D = discriminator(self.opts.batch_size,self.opts.patch_num_point)
      # G = generator(self.opts.batch_size,self.opts.num_point)
      D = Discriminator(is_training=True)
      G = Generator(self.opts, is_training=True)
      # g_checkpoint = tf.train.Checkpoint(self.G, optimizer=self.g_optimizer)
      self.d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_d, beta_1=self.opts.beta)
      self.g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_g, beta_1=self.opts.beta)

      # self.d_optimizer = tf.optimizers.SGD(learning_rate=learning_rate_d)
      # self.g_optimizer = tf.optimizers.SGD(learning_rate=learning_rate_g, momentum=self.opts.beta)


      d_checkpoint = tf.train.Checkpoint(D, optimizer=self.d_optimizer)
      g_checkpoint = tf.train.Checkpoint(G, optimizer=self.g_optimizer)
      # ckpt管理器
      ckpt_manager_g = tf.train.CheckpointManager(g_checkpoint, self.opts.save_g, max_to_keep=20)
      ckpt_manager_d = tf.train.CheckpointManager(d_checkpoint, self.opts.save_d, max_to_keep=20)
      if self.opts.restore == True:
          g_checkpoint.restore(tf.train.latest_checkpoint(self.opts.save_g))
          #g_checkpoint.restore(self.opts.save_g + '/' + 'ckpt-115')
          d_checkpoint.restore(tf.train.latest_checkpoint(self.opts.save_d))
          #d_checkpoint.restore(self.opts.save_d + '/' + 'ckpt-42')

      for epoch in range(1, self.opts.training_epoch):
          patch = 0
          # uni = None
          # CD = []
          # HD = []
          # EMD = []
          for batch_idx in range(fetchworker.num_batches):
              input_x, input_y,radius = fetchworker.fetch()
              # train 判别器
              # for i in range(self.opts.d_update):

              with tf.GradientTape() as tape:
                  d_loss, real_loss, fake_loss = self.D_loss(G,D,input_x,input_y, patch, radius)
              grads = tape.gradient(d_loss, D.trainable_variables)
              grad = np.array(grads[0][0][0][0][0])
              if np.isnan(grad):
                  print('nan')
                  continue
              self.d_optimizer.apply_gradients(zip(grads, D.trainable_variables))
              # CD.append(cd)
              # HD.append(hd_value)
              # self.d_optimizer.minimize(loss = d_loss, var_list = D.trainable_variables, tape = d_tape)
              # train 生成器
              for i in range(self.opts.gen_update):
                  with tf.GradientTape() as tape:
                      g_loss = self.G_loss(G,D,input_x, input_y, radius,batch_idx,i)
                  grads = tape.gradient(g_loss, G.trainable_variables)
                  grad = np.array(grads[0][0][0][0][0])
                  if np.isnan(grad):
                      print('nan')
                      break
                  self.g_optimizer.apply_gradients(zip(grads, G.trainable_variables))
                  # self.g_optimizer.minimize(loss=g_loss, var_list=G.trainable_variables, tape=g_tape)
              # EMD.append(emd)
              patch += 1
              # un_loss = np.reshape(un_loss,(1,5))
              # if uni is not None:
              #     uni = np.concatenate((uni, un_loss),axis=0)
              # else:
              #     uni = un_loss
              if patch % self.opts.steps_per_print == 0:
                  print('epoch:',epoch, 'patch:', patch, 'D_loss:', float(d_loss), 'G_loss:', float(g_loss),'real_loss:', float(real_loss), 'fake_loss:', float(fake_loss))#, 'uniform_loss:', float(self.uniform_loss))
              if (patch % self.opts.epoch_per_save) == 0:
                  ckpt_manager_g.save()
                  ckpt_manager_d.save()
          # CD = tf.add_n(CD)/250
          # HD = tf.add_n(HD)/250
          # EMD = tf.add_n(EMD) / 250
          # uni = tf.reduce_mean(uni, axis=0)
          # print(CD, HD, uni, EMD)
      fetchworker.shutdown()

  def patch_prediction(self, patch_point):
      # normalize the point clouds
      patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
      # patch_point = np.expand_dims(patch_point, axis=0)
      # patch_point = tf.tile(patch_point, [4,1,1])
      pred = self.pred(patch_point)
      pred = centroid + pred * furthest_distance
      return pred

  def pc_prediction(self, pc):
      ## get patch seed from farthestsampling
      # points = np.expand_dims(pc,axis=0)
      seed_num = int(pc.shape[0] / self.opts.num_point * 3)#self.opts.patch_num_ratio)

      ## FPS sampling
      sampler = pc_util.FarthestSampler()
      farthest_pts,seed = sampler(pc, seed_num) #seed:n
      seed_list = seed[:seed_num]
      print("number of patches: %d" % len(seed_list))
      input_list = []
      up_point_list=[]

      patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.num_point)
      patches_point = []
      self.opts.batch_size = 6
      for point in tqdm(patches, total=len(patches)):
            patches_point.append(point)
            if len(patches_point)==self.opts.batch_size:
                input_list.append(point)
                patches_point =  np.array(patches_point)
                up_point = self.patch_prediction(patches_point)
                for i in range(self.opts.batch_size):
                    up_point_list.append(up_point[i])
                patches_point = []
      return input_list, up_point_list

  def test(self):
      learning_rate_d = self.opts.base_lr_d
      learning_rate_g = self.opts.base_lr_g

      # D = discriminator(self.opts.batch_size, self.opts.patch_num_point)
      # G = generator(self.opts.batch_size, self.opts.num_point)
      D = Discriminator(is_training=True)
      G = Generator(self.opts, is_training=True)
      self.d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_d, beta_1=self.opts.beta)
      self.g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_g, beta_1=self.opts.beta)

      d_checkpoint = tf.train.Checkpoint(D, optimizer=self.d_optimizer)
      g_checkpoint = tf.train.Checkpoint(G, optimizer=self.g_optimizer)
      g_checkpoint.restore(tf.train.latest_checkpoint(self.opts.save_g))
      d_checkpoint.restore(tf.train.latest_checkpoint(self.opts.save_d))
      #d_checkpoint.restore(self.opts.save_d + '/' + 'ckpt-100')
      #g_checkpoint.restore(self.opts.save_g + '/' + 'ckpt-100')
      self.pred = G
      for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
          self.pred = G(self.pred)
      samples = glob(self.opts.test_data)
      #point = pc_util.load(samples[0])

      for point_path in samples:
          pc = pc_util.load(point_path)[:, :3]
          
          num_point = pc.shape[0]
          out_point_num = int(num_point * self.opts.up_ratio)
          pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)
          if self.opts.jitter:
              pc = pc_util.jitter_perturbation_point_cloud(pc[np.newaxis, ...], sigma=self.opts.jitter_sigma,
                                                             clip=self.opts.jitter_max)
              pc = pc[0, ...]
          input_list, pred_list = self.pc_prediction(pc)

          pred_pc = np.concatenate(pred_list, axis=0)
          pred_pc = (pred_pc * furthest_distance) + centroid

          pred_pc = np.reshape(pred_pc, [-1, 3])
          sampler = pc_util.FarthestSampler()
          pred_pc, idx = sampler(pred_pc, out_point_num)
          path = os.path.join(self.opts.out_folder[:], point_path.split('/', 7)[-1])
          print(path)
          np.savetxt(path, pred_pc, fmt='%.6f')

