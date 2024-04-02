import argparse
import os

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test',help="train/test")
parser.add_argument('--log_dir', default='/home/aiguo/pu_gan3.0_crt/log')
parser.add_argument('--data_dir', default='/home/aiguo/pu_gan3.0_crt/data')
parser.add_argument('--augment', type=str2bool, default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--more_up', type=int, default=2) #Generator
parser.add_argument('--training_epoch', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=28)
parser.add_argument('--use_non_uniform', type=str2bool, default=True)
parser.add_argument('--jitter', type=str2bool, default=False)
parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
parser.add_argument('--up_ratio', type=int, default=4) #test/Generator
parser.add_argument('--num_point', type=int, default=256)
parser.add_argument('--patch_num_point', type=int, default=1024) #test/Generator
parser.add_argument('--patch_num_ratio', type=int, default=3) #test
parser.add_argument('--base_lr_d', type=float, default=0.000003) #optimizer
parser.add_argument('--base_lr_g', type=float, default=0.00003) #optimizer
parser.add_argument('--beta', type=float, default=0.9) #optimizer
parser.add_argument('--lr_decay_steps_d', type=int, default=8570) #optimizer
parser.add_argument('--lr_decay_steps_g', type=int, default=8570) #optimizer
parser.add_argument('--lr_decay_rate', type=float, default=0.7) #optimizer
parser.add_argument('--lr_clip', type=float, default=1e-6) #optimizer
parser.add_argument('--steps_per_print', type=int, default=1) #train
parser.add_argument('--visulize', type=str2bool, default=False)
parser.add_argument('--steps_per_visu', type=int, default=100)
parser.add_argument('--epoch_per_save', type=int, default=857) #train
parser.add_argument('--use_repulse', type=str2bool, default=True) #build_model repulsion_loss
parser.add_argument('--repulsion_w', default=1.0, type=float, help="repulsion_weight") #build_model repulsion_loss
parser.add_argument('--fidelity_w', default=100, type=float, help="fidelity_weight") #build_model dis_loss
parser.add_argument('--uniform_w', default=10.0, type=float, help="uniform_weight") #build_model uniform_loss
parser.add_argument('--gan_w', default=0.5, type=float, help="gan_weight") #build_model G_gan_loss
parser.add_argument('--gen_update', default=2, type=int, help="gen_update") #train
FLAGS = parser.parse_args()