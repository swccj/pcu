import tensorflow as tf
from model import Model
from configs import FLAGS
from datetime import datetime
import os
import logging
import pprint

#os.environ['CUDA_VISIBLE_DEVICES']='3'
pp = pprint.PrettyPrinter()

FLAGS.save_g = os.path.join(FLAGS.log_dir, 'armadillo_g')
FLAGS.save_d = os.path.join(FLAGS.log_dir, 'armadillo_d')
#FLAGS.out_folder = os.path.join(FLAGS.data_dir,'test/output')
FLAGS.test_data = os.path.join(FLAGS.data_dir, 'test/0.005/*.xyz')
if FLAGS.phase=='train':
    FLAGS.train_file = os.path.join(FLAGS.data_dir, 'train/PUGAN_poisson_256_poisson_1024.h5')
else:
    FLAGS.out_folder = os.path.join(FLAGS.data_dir,'test/0.005_out')
    if not os.path.exists(FLAGS.out_folder):
        os.makedirs(FLAGS.out_folder)
    print('test_data:',FLAGS.test_data)
    print('test_data:',FLAGS.out_folder)


model = Model(FLAGS)
run_config = tf.compat.v1.ConfigProto()
run_config.gpu_options.allow_growth = True
if FLAGS.phase == 'train':
    model.train()
    #model.test()
    # model.save('C:/Users/15463/Desktop/1/venv/model/pu_gan')
else:
    model.test()




