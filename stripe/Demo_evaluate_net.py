"""
This file evaluates a trained network and create plots of it's performance.

It reads the configuration file config_val.ini. If the script is called
without any command line arguments, the network with runner_id specified in 
config_val.ini will be evaluated. It is possible it specify the runner_id of
the network you would like to evaluate as a command line argument. Call the 
script as follows
```
python Demo_evaluate_net.py runner_id
```
where `runner_id` is an integer specifying the network. In this case 
all other options than the runner_id in the config_val.ini file will be used
for evaluation.
"""

import matplotlib.pyplot as plt;
from scipy.io import savemat, loadmat; 
from scipy.io import savemat, loadmat;
import sys;
import os;
from os.path import join;
import numpy as np;
import tensorflow as tf;
from Config_handler import Config_handler;
import configparser;
import functools;
from Data_loader import Data_loader;

dest_plots = './plots'
# Read the validation and training config
val_config_fname = 'config_val.ini';
config_val = configparser.ConfigParser()
config_val.read(val_config_fname)

if len(sys.argv) == 1:
    runner_id = int(config_val['VAL']['runner_id']);
else:
    runner_id = int(sys.argv[1]);
#runner_id         = int(config_val['VAL']['runner_id']);
dest_model        = config_val['VAL']['dest_model'];

dir_name = 'run_%03d' % (runner_id);
run_dir  = join(dest_model, dir_name);
train_config_fname = join(run_dir, 'config.ini')
config_train = configparser.ConfigParser();
config_train.read(train_config_fname);
run_dir_as_mod = run_dir.replace('/', '.');
exec_str = 'from %s.simple_nn import *;' % (run_dir_as_mod);
print(exec_str)
exec(exec_str);

# Create Config Handler

ch = Config_handler(config_train, config_val);

# Experiment
#runner_id        = ch.runner_id
ch.runner_id = runner_id
epoch_nbr        = ch.epoch_nbr

# DATASET
n             = ch.n;
line_width    = ch.line_width;
a             = ch.a;
a_low_test    = ch.a_low_test;

train_size    = int( 2*(n - line_width + 1) );
test_size     = ch.test_size;

# SETUP
use_gpu       = ch.use_gpu;
compute_node  = ch.compute_node;
ckpt_dir_name = ch.ckpt_dir;
model_name    = ch.model_name;
print_every   = ch.print_every
save_every    = ch.save_every;
fsize         = ch.fsize;

nbr_epochs    = ch.nbr_epochs;
network_str   = ch.network_str;
prec          = ch.prec;
loss_type     = ch.loss;


print(ch);


if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node);
    dev_name = "/device:GPU:0"
else:
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"; 
    dev_name = "/device:CPU:%d" % (compute_node);


data_it = ph_test_time(prec=prec,n=n, dev_name=dev_name);
batch_x = data_it['batch_x'];
batch_label = data_it['batch_label'];

network = eval(network_str);
net     = network(batch_x, prec=prec, dev_name=dev_name)
pred    = net['pred'];

test_loss   = test_time_loss(pred, batch_label, loss_type, dev_name=dev_name);
accuracy    = test_loss['accuracy'];
batch_loss  = test_loss['batch_loss'];


saver = tf.train.Saver()

ckpt_dir = join(run_dir, ckpt_dir_name);

ckpt_nbr = 0;
if epoch_nbr == -1:
    ckpt_nbr = nbr_epochs;
else:
    ckpt_nbr = save_every*epoch_nbr;
ckpt_name = join( ckpt_dir, "%s-%d" % (model_name, ckpt_nbr) );

print('Runner: %d' % runner_id)
print('\n------- Trained with ------');
print('n = %d' % n);
print('line_width = %d' % line_width);
print('a             = %g' % a);
print('a_low_test    = %g' % a_low_test);
print('---------------------------');


with tf.Session() as sess:
    saver.restore(sess, ckpt_name) 
   
    print('Size of test set: %d' % test_size);
    for i in range(1, 3):
        base = 10**(-i-1);
        for k in range(9,0,-1):
            a_high = (k+1)*base;
            a_low = k*base;

            data_loader = Data_loader(n=n,a=a,line_width=line_width);
            [x_test, label_test]  = data_loader.load_data_test(test_size, a_low=a_low, a_high=a_high); 
            
            [x_test_false, label_test_false]  = data_loader.load_data_test_false(test_size, a_low=a_low, a_high=a_high); 
            
            acc, nbr_of_ones = sess.run([accuracy, batch_loss], 
                                    feed_dict={batch_x: x_test,
                                    batch_label: label_test});

            acc_false, nbr_of_ones_false = sess.run([accuracy, batch_loss], 
                                    feed_dict={batch_x: x_test_false,
                                    batch_label: label_test_false});
            if (abs(a_high -a) > 1e-10): 
                print('a ∈ [%9f, %9f], True acc: %4.1f, False_acc: %4.1f' % (a_low, a_high, 100*acc, 100*acc_false));
            else:
                print('')
                print('a ∈ [%9f, %9f], True acc: %4.1f, False_acc: %4.1f' % (a_low, a_high, 100*acc, 100*acc_false));
                print('')


