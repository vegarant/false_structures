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

n             = ch.n;
line_width    = ch.line_width;
a             = ch.a;
a_low_test    = ch.a_low_test;

train_size = int( 2*(n - line_width + 1) );
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
plot_training            = ch.plot_training
plot_accuracy            = ch.plot_accuracy

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

data_loader = Data_loader(n=n, a=a, line_width=line_width);

print('Creating data...');
[x_train, label_train]  = data_loader.load_data_train(shuffle=True); 

[x_val, label_val] = data_loader.load_data_test(test_size, a_low=a_low_test, 
                                                a_high=a); 
[x_val_false, label_val_false] = data_loader.load_data_test_false(test_size, 
                                                   a_low=a_low_test, a_high=a); 


print('Data created');

data_it = create_data_iterator(train_size, n, shuffle, prec=prec);
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
os.system('rm -f ' + join(ckpt_dir, '*'));

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

train_loss_arr     = np.zeros(nbr_epochs//print_every);
val_loss_arr       = np.zeros(nbr_epochs//print_every);
val_loss_false_arr = np.zeros(nbr_epochs//print_every);

train_acc_arr     = np.zeros(nbr_epochs//print_every);
val_acc_arr       = np.zeros(nbr_epochs//print_every);
val_acc_false_arr = np.zeros(nbr_epochs//print_every);

train_init_op = data_it['train_init_op'];
val_init_op = data_it['val_init_op'];
test_init_op = data_it['test_init_op'];
data_x = data_it['data_x'];
data_label = data_it['data_label'];
ph_batch_size = data_it['ph_batch_size'];



with tf.Session() as sess:
    sess.run(init);
    sess.run(train_init_op, feed_dict={data_x: x_train, 
                                       ph_batch_size: batch_size,
                                       data_label: label_train});

    tvars = tf.trainable_variables();
    tvars_vals = sess.run(tvars);

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
                                       ph_batch_size: test_size,
                                       data_label: label_val});

            val_loss, val_acc = sess.run([loss, accuracy]);

            sess.run(test_init_op, feed_dict={data_x: x_val_false, 
                                       ph_batch_size: test_size,
                                       data_label: label_val_false});

            val_loss_false, val_acc_false = sess.run([loss, accuracy]);
            
            sess.run(train_init_op, feed_dict={data_x: x_train,
                                       ph_batch_size: batch_size,
                                       data_label: label_train});

            print('  Trl: %g, Tra: %g, Vl: %g, Va: %g, Vfl:%g, Vfa: %g' %\
                    (train_loss, train_acc, val_loss, val_acc, val_loss_false, val_acc_false));

            train_loss_arr[epoch//print_every]       = train_loss;
            val_loss_arr[epoch//print_every]         = val_loss;
            val_loss_false_arr[epoch//print_every]   = val_loss_false;
            
            train_acc_arr[epoch//print_every]       = train_acc;
            val_acc_arr[epoch//print_every]         = val_acc;
            val_acc_false_arr[epoch//print_every]   = val_acc_false;


    sess.run(val_init_op, feed_dict={data_x: x_val, 
                                     ph_batch_size: test_size,
                                     data_label: label_val});

    val_loss, val_acc = sess.run([loss, accuracy]);

    sess.run(test_init_op, feed_dict={data_x: x_val_false,
                               ph_batch_size: test_size,
                               data_label: label_val_false});

    val_loss_false, val_acc_false = sess.run([loss, accuracy]);
    
    sess.run(train_init_op, feed_dict={data_x: x_train,
                               ph_batch_size: batch_size,
                               data_label: label_train});

    train_loss, train_acc = sess.run([loss, accuracy]);
    
    print('  Trl: %g, Tr_acc: %g, Vl: %g, V_acc: %g, Vfl:%g, Vfa: %g' %\
            (train_loss, train_acc, val_loss, val_acc, val_loss_false, val_acc_false));
    
    train_loss_arr[-1]     = train_loss;
    val_loss_arr[-1]       = val_loss;
    val_loss_false_arr[-1] = val_loss_false;
    
    train_acc_arr[-1]     = train_acc;
    val_acc_arr[-1]       = val_acc;
    val_acc_false_arr[-1] = val_acc_false;

    save_path = saver.save(sess, export_model, global_step = nbr_epochs);

    tvars_vals = sess.run(tvars);
    
    print('\nSize and name of variable');
#    for var, val in zip(tvars, tvars_vals):
#            print(var.name, val.shape) 
#            print(sess.run(var));

    print("\n\nModel saved in path: %s\n\n" % save_path)  
    savemat(join(run_dir, 'train_data.mat'), mdict={ 'val_loss_arr': val_loss_arr,
                                                     'train_loss_arr': train_loss_arr})

    print('\nRunner: %d\n' % (count));

    os.system('rm -f ' + join(plot_dir, 'plot_*'));

    if plot_training:    
        print('Creating: plot_loss.png');
        fig = plt.figure();
        plt.semilogy(epoch_arr, train_loss_arr, label='Train'); 
        plt.semilogy(epoch_arr, val_loss_arr,   label='Val'); 
        plt.semilogy(epoch_arr, val_loss_false_arr,   label='Val false'); 
        plt.xlabel('Number of epochs', fontsize=fsize);
        plt.ylabel('Loss', fontsize=fsize);
        plt.legend(fontsize=fsize);
        plt.savefig(join(plot_dir, 'plot_loss.png'));

    if plot_accuracy:    
        print('Creating: plot_accuracy.png');
        fig = plt.figure();
        plt.plot(epoch_arr, train_acc_arr, label='Train'); 
        plt.plot(epoch_arr, val_acc_arr,   label='Validation'); 
        plt.plot(epoch_arr, val_acc_false_arr,   label='Flipped val');
        plt.xlabel('Number of epochs', fontsize=fsize);
        plt.ylabel('Accuracy', fontsize=fsize);
        plt.legend(fontsize=fsize);
        plt.savefig(join(plot_dir, 'plot_accuracy.png'));





