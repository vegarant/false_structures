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
    runner_id         = int(config_val['VAL']['runner_id']);
else:
    runner_id = int(sys.argv[1]);
print(sys.argv)
#runner_id         = int(config_val['VAL']['runner_id']);
dest_model        = config_val['VAL']['dest_model'];

run_name = 'run_%03d' % (runner_id);
plot_name = 'plot_%03d' % (runner_id);
run_dir  = join(dest_model, run_name);
plot_dir  = join(dest_model, plot_name);
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
ch.runner_id = runner_id;
epoch_nbr        = ch.epoch_nbr

# DATASET
a             = ch.a;
K             = ch.K;
eps           = ch.eps;
delta         = ch.delta;
add_delta     = ch.add_delta;
data_chooser  = ch.data_chooser;
draw_dist     = ch.draw_dist;
data_size     = ch.data_size;
train_size    = ch.train_size;
val_size      = ch.val_size;
test_size     = ch.test_size;

# SETUP
use_gpu       = ch.use_gpu;
compute_node  = ch.compute_node;
ckpt_dir_name = ch.ckpt_dir;
model_name    = ch.model_name;
print_every   = ch.print_every
save_every    = ch.save_every;
fsize         = 20; ch.fsize;
lwidth        = 5;

nbr_epochs    = ch.nbr_epochs;
network_str   = ch.network_str;
prec          = ch.prec;
loss_type     = ch.loss;
print('loss_type: ', loss_type)
# Plot            
plot_first_comp          = ch.plot_first_comp
plot_first_comp_general  = ch.plot_first_comp_general
plot_second_comp         = ch.plot_second_comp
plot_second_comp_general = ch.plot_second_comp_general
general_c                = ch.general_c



print(ch);


if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node);
    dev_name = "/device:GPU:0"
else:
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"; 
    dev_name = "/device:CPU:%d" % (compute_node);


data_it = ph_test_time(prec=prec, dev_name=dev_name);
batch_x = data_it['batch_x'];
batch_label = data_it['batch_label'];

network = eval(network_str);
net     = network(batch_x, prec=prec, dev_name=dev_name)
pred    = net['pred'];

test_loss   = test_time_loss(pred, batch_label, loss_type,dev_name=dev_name);
accuracy    = test_loss['accuracy'];
batch_loss  = test_loss['batch_loss'];

data_loader = Data_loader(a=a, K=K, eps=eps, delta=delta);

print('data_chooser: ', data_chooser)
[x_test, label_test]  = data_loader.load_data_test(train_size, add_delta=add_delta, data_chooser=data_chooser, dist=draw_dist);

saver = tf.train.Saver()

ckpt_dir = join(run_dir, ckpt_dir_name);

ckpt_nbr = 0;
if epoch_nbr == -1:
    ckpt_nbr = nbr_epochs;
else:
    ckpt_nbr = save_every*epoch_nbr;   
ckpt_name = join( ckpt_dir, "%s-%d" % (model_name, ckpt_nbr) );


with tf.Session() as sess:
    saver.restore(sess, ckpt_name) 
    
    l1 = sess.run(accuracy, feed_dict={batch_x: x_test,
                                        batch_label: label_test});

    l2 = sess.run(batch_loss,  feed_dict={batch_x: x_test,
                                          batch_label: label_test});
    
    l3 = sess.run(tf.reduce_mean(batch_label),  
                  feed_dict={batch_label: label_test});
    l4 = sess.run(tf.reduce_sum(batch_label),  
                  feed_dict={batch_label: label_test});

    nbr_of_labels = len(label_test);
    print('\n\n------------------------------------------------')
    print('Add delta:        %s' % (add_delta));
    print('Accuracy:         %g' % (l1) );
    print('Batch loss:       %g' % (l2) );
    print('Fraction of ones: %g' % (l3) );
    print('Number of ones:   %g of %d' % (l4, nbr_of_labels) );
    print('------------------------------------------------\n\n')
    
    f = lambda x: data_loader.target_function(x);
    N1 = 501;
    N2 = 151;
    N = N1 + N2;
    b = data_loader.b;
    print('b = ', b);
    print('\nRunner: %d\n' % (runner_id));

    #os.system('rm -f ' + join(plot_dir, 'plot_*'));


    if plot_first_comp:
        print('Creating: plot_first_comp1.png');
        t = np.linspace(b,1-1e-8, N1);

        linear_input = np.zeros([N1,2]);
        linear_input[:,0] = t;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(x)', linewidth=lwidth);
        plt.plot(t, net_out, label='network', linewidth=lwidth);
        plt.xticks(fontsize=fsize);
        plt.yticks(fontsize=fsize);
#        plt.xlabel('First component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.00, 1, 0.1), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_first_comp1.png'), bbox_inches='tight');
    
    if plot_first_comp_general:
        print('Creating: plot_first_comp_general1.png');
        if general_c > 1:
            t1 = np.linspace(b,1, N1);
            t2 = np.linspace(1, general_c, N2);
            t  = np.concatenate((t1,t2)); 
        else:
            t = np.linspace(general_c, 1-1e-8,N);
        linear_input = np.zeros([N,2]);
        linear_input[:,0] = t;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(x)', linewidth=lwidth);
        plt.plot(t, net_out, label='network', linewidth=lwidth);
        plt.xticks(fontsize=fsize);
        plt.yticks(fontsize=fsize);
#        plt.xlabel('First component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.00, 1, 0.1), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_first_comp_general1.png'), bbox_inches='tight');

    if plot_second_comp:
        print('Creating: plot_second_comp1.png');
        t = np.linspace(b,1-1e-8, N1);
        lab = f(t);
        
        linear_input = np.zeros([N1,2]);
        linear_input[:,0] = t;
        linear_input[:,1] = delta*lab;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(x)', linewidth=lwidth);
        plt.plot(t, net_out, label='network', linewidth=lwidth);
        plt.xticks(fontsize=fsize);
        plt.yticks(fontsize=fsize);
        #plt.xlabel('Second component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.00, 1, 0.1), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_second_comp1.png'), bbox_inches='tight');
    
    if plot_second_comp_general:
        print('Creating: plot_second_comp_general1.png');
        if general_c > 1:
            t1 = np.linspace(b,1, N1);
            t2 = np.linspace(1, general_c, N2);
            t  = np.concatenate((t1,t2)); 
        else:
            t = np.linspace(general_c, 1-1e-8,N);
        lab = f(t);

        linear_input = np.zeros([N,2]);
        
        linear_input[:,0] = t;
        linear_input[:,1] = delta*lab;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(x)', linewidth=lwidth);
        plt.plot(t, net_out, label='network', linewidth=lwidth);
#        plt.xlabel('Second component', fontsize=fsize);
        plt.xticks(fontsize=fsize);
        plt.yticks(fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.00, 1, 0.1), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_second_comp_general1.png'), bbox_inches='tight');



