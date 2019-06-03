"""
This file reads the configuration file config.ini and starts to train a
network based on the settings in this.

Calling the script like 
```
python Demo_train_network.py X
```
where X is an integer will start the network to train using the configuration
file configX.ini.
"""

import matplotlib.pyplot as plt;
from scipy.io import savemat, loadmat; 
from scipy.io import savemat, loadmat;
from nn_tools import read_count;
from os.path import join;
import sys;
import os;
import numpy as np;
import tensorflow as tf;
from Data_loader import Data_loader, create_data_iterator;
from Config_handler import Config_handler;
import shutil;
import configparser;
import functools;

if len(sys.argv) == 1:
    config_filename = './config.ini';
else:
    config_nbr = int(sys.argv[1]);
    config_filename = './config%d.ini' % (config_nbr);


config = configparser.ConfigParser()
config.read(config_filename)

ch = Config_handler(config);
print(ch)

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

use_gpu       = ch.use_gpu;
compute_node  = ch.compute_node;
tf_log_level  = ch.tf_log_level;
prec          = ch.prec;
dest_model    = ch.dest_model;
print_every   = ch.print_every;
save_every    = ch.save_every;
ckpt_dir_name = ch.ckpt_dir;
model_name    = ch.model_name;
counter_path  = ch.counter_path;
fsize         = ch.fsize;
runner_id     = ch.runner_id;
use_run_count = ch.use_run_count;
            
# Plot            
plot_first_comp          = ch.plot_first_comp
plot_first_comp_general  = ch.plot_first_comp_general
plot_second_comp         = ch.plot_second_comp
plot_second_comp_general = ch.plot_second_comp_general
plot_f_and_data          = ch.plot_f_and_data
plot_f_and_data2         = ch.plot_f_and_data2
plot_data_dist           = ch.plot_data_dist
plot_training            = ch.plot_training
general_c                = ch.general_c

nbr_epochs    = ch.nbr_epochs;
batch_size    = ch.batch_size;
shuffle       = ch.shuffle;
optim         = ch.optim;
network_str   = ch.network_str;
loss_type     = ch.loss;

lr_dict = ch.lr_dict;


if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node);
    dev_name = "/device:GPU:0"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"; 
    dev_name = "/device:CPU:%d" % (compute_node);
if (tf_log_level+1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '%d' % (tf_log_level);

data_loader = Data_loader(a=a, K=K, eps=eps, delta=delta);

print('Creating data...');
[x_train, label_train]  = data_loader.load_data_train(train_size, add_delta=add_delta, data_chooser=data_chooser, dist=draw_dist);
[x_val, label_val]      = data_loader.load_data_val(val_size, add_delta=add_delta, data_chooser=data_chooser, dist=draw_dist);
print('Data created');

data_it = create_data_iterator(train_size, shuffle, prec=prec);
batch_x = data_it['batch_x'];
batch_label = data_it['batch_label'];

# Save everything
if use_run_count:
    count = read_count(count_path=counter_path);
else:
    count = runner_id;

print("""
#########################################################
###              Saving as run: %-5d                 ###
#########################################################
""" % (count));
b = data_loader.b;
a = data_loader.a;
print('b: %g' % data_loader.b);
print('max eps: %g' % (b*b/(2*(a-b))) )

if not os.path.isdir(dest_model):
    os.mkdir(dest_model);

dir_name = 'run_%03d' % (count);
dir_name_plot = 'plot_%03d' % (count);
run_dir = join(dest_model, dir_name);
plot_dir = join(dest_model, dir_name_plot);
if not os.path.isdir(run_dir):
    os.mkdir(run_dir);
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir);
ckpt_dir = join(run_dir, ckpt_dir_name)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir);

export_model = join(ckpt_dir, model_name);

# Copy simple_nn.py and config file, and load the newly moved 
# simple_nn file.
shutil.copyfile('./simple_nn.py', join(run_dir, 'simple_nn.py')); 
shutil.copyfile(config_filename, join(run_dir, 'config.ini')); 
shutil.copyfile('./simple_nn.py', join(plot_dir, 'simple_nn.py')); 
shutil.copyfile(config_filename, join(plot_dir, 'config.ini')); 

init_py_file = join('%s' % (run_dir), '__init__.py');
open(init_py_file, 'a');
run_dir_as_mod = run_dir.replace('/', '.');
exec_str = 'from %s.simple_nn import *;' % (run_dir_as_mod);
exec(exec_str);

# Load the network architechture  
network = eval(network_str);
net = network(batch_x, dev_name=dev_name, prec=prec);
pred = net['pred'];

test_loss   = test_time_loss(pred, batch_label, loss_type, dev_name=dev_name);
accuracy    = test_loss['accuracy'];

train = generate_training(pred, batch_label, optim, loss_type, lr_dict=lr_dict, batch_size=None, dev_name=dev_name, prec=prec);

loss = train['loss'];
optimizer = train['optimizer'];

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

init = tf.global_variables_initializer();

epoch_arr = np.linspace(0, nbr_epochs, nbr_epochs//print_every);
train_loss_arr = np.zeros(nbr_epochs//print_every);
val_loss_arr   = np.zeros(nbr_epochs//print_every);


train_init_op = data_it['train_init_op'];
val_init_op = data_it['val_init_op'];
data_x = data_it['data_x'];
data_label = data_it['data_label'];
ph_batch_size = data_it['ph_batch_size'];
   


with tf.Session() as sess:
    sess.run(init);
    sess.run(train_init_op, feed_dict={data_x: x_train, 
                                       ph_batch_size: batch_size,
                                       data_label: label_train});

    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    
    print('\nSize and name of variable');
    for var, val in zip(tvars, tvars_vals):
            print(var.name, val.shape)  # Prints the name of the variable alongside its value.
    print('\n\n');
    
    m = train_size//batch_size;

    for epoch in range(nbr_epochs):
        # print current epoch
        sys.stdout.write("\rEpoch: %d/%d" % (epoch+1, nbr_epochs))
        sys.stdout.flush()
        for i in range(m):
            out = sess.run(optimizer);
        
        if not (epoch % save_every):
            save_path = saver.save(sess, export_model, global_step=epoch)

        if not (epoch % print_every):

            train_loss, train_acc = sess.run([loss, accuracy]);
            
            sess.run(val_init_op, feed_dict={data_x: x_val, 
                                       ph_batch_size: val_size,
                                       data_label: label_val});

            val_loss, val_acc = sess.run([loss, accuracy]);

            sess.run(train_init_op, feed_dict={data_x: x_train,
                                       ph_batch_size: batch_size,
                                       data_label: label_train});
            
            print('  Train loss: %g, Train acc: %g, val loss: %g, val acc: %g' %\
                    (train_loss, train_acc, val_loss, val_acc));
            
            train_loss_arr[epoch//print_every] = train_loss;
            val_loss_arr[epoch//print_every]   = val_loss;

    train_loss, train_acc = sess.run([loss, accuracy]);

    sess.run(val_init_op, feed_dict={data_x: x_val, 
                                     ph_batch_size: val_size,
                                     data_label: label_val});

    val_loss, val_acc = sess.run([loss, accuracy]);

    sess.run(train_init_op, feed_dict={data_x: x_train,
                               ph_batch_size: batch_size,
                               data_label: label_train});

    print('  Train loss: %g, Train acc: %g, val loss: %g, val acc: %g' %\
              (train_loss, train_acc, val_loss, val_acc));
    train_loss_arr[-1] = train_loss;
    val_loss_arr[-1]   = val_loss;


    save_path = saver.save(sess, export_model, global_step = nbr_epochs);


    print("\n\nModel saved in path: %s\n\n" % save_path)  
    savemat(join(run_dir, 'train_data.mat'), mdict={ 'val_loss_arr': val_loss_arr,
                                                     'train_loss_arr': train_loss_arr})
    f = lambda x: data_loader.target_function(x);
    N1 = 501;
    N2 = 151;
    N = N1 + N2;
    b = data_loader.b;
    print('b = ', b);
    print('\nRunner: %d\n' % (count));

    os.system('rm -f ' + join(plot_dir, 'plot_*'));

    if plot_training:    
        print('Creating: plot_loss.png');
        fig = plt.figure();
        plt.semilogy(epoch_arr, train_loss_arr, label='Train'); 
        plt.semilogy(epoch_arr, val_loss_arr,   label='Validation'); 
        plt.xlabel('Number of epochs', fontsize=fsize);
        plt.ylabel('Loss', fontsize=fsize);
        plt.legend(fontsize=fsize);
        plt.savefig(join(plot_dir, 'plot_loss.png'));

    if plot_first_comp:
        print('Creating: plot_first_comp.png');
        t = np.linspace(b,1-1e-8, N1);

        linear_input = np.zeros([N1,2]);
        linear_input[:,0] = t;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.xlabel('First component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_first_comp.png'));
    
    if plot_first_comp_general:
        print('Creating: plot_first_comp_general.png');
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
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.xlabel('First component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_first_comp_general.png'));

    if plot_second_comp:
        print('Creating: plot_second_comp.png');
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
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.xlabel('Second component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_second_comp.png'));
    
    if plot_second_comp_general:
        print('Creating: plot_second_comp_general.png');
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
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.xlabel('Second component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_second_comp_general.png'));

    if plot_f_and_data:
        print('Creating: plot_f_and_data.png');
        t = np.linspace(b,1-1e-8, N1);
        x = x_train[:,0];

        linear_input = np.zeros([N1,2]);
        linear_input[:,0] = t;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.plot(x, f(x), '.');
        plt.xlabel('First component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_f_and_data.png'));
    
    if plot_f_and_data2:
        print('Creating: plot_f_and_data2.png');
        t = np.linspace(b,1-1e-8, N1);
        lab = f(t);
        x = x_train[:,0];
        
        linear_input = np.zeros([N1,2]);
        linear_input[:,0] = t;
        linear_input[:,1] = delta*lab;
        if loss_type.lower() == 'sigmoid_cross_entropy':
            net_out = sess.run(tf.math.round(tf.nn.sigmoid(pred)), feed_dict = {batch_x: linear_input});
        elif loss_type.lower() == 'mean_squared_error':
            net_out = sess.run(tf.math.round(pred), feed_dict = {batch_x: linear_input});

        fig = plt.figure();
        plt.plot(t,f(t), label='f(t)');
        plt.plot(t, net_out, label='network');
        plt.plot(x, f(x), '.');
        plt.xlabel('Second component', fontsize=fsize);
        plt.legend(fontsize=fsize, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2);
        plt.savefig(join(plot_dir, 'plot_f_and_data2.png'));


    if plot_data_dist:
        print('Creating: plot_datadist.png');

        k_values = np.linspace(K+1, 1, K+1);
        fig = plt.figure();
        plt.hist(x_train, bins=(a/k_values));
        plt.savefig(join(plot_dir, 'plot_datadist.png'));


