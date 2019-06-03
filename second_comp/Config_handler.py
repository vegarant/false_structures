import configparser;
import functools;
import os;
import tensorflow as tf;


class Config_handler:
    """
Class to handle the configuration file. 

The main purpose of this class is to convert all the all the strings in the 
configuration files to the right format.
"""

    def __init__(self, config_train, config_val=None):
        """
The idea of this initialization is to read and set the right config options.

If you only provide config_train, then those options are used. Otherwise 
both options from config_val and config_train are used, with a 
preference for the options specified in config_val. 
        """
        self.config_train = config_train;
        self.config_val = config_val;

        # Special validatation options
        if config_val is not None:

            self.runner_id         = int(config_val['VAL']['runner_id']);
            self.dest_model        = config_val['VAL']['dest_model'];
            self.epoch_nbr         = int(config_val['VAL']['epoch_nbr']);
            self.use_gpu           = eval(config_val['VAL']['use_gpu']);
            self.compute_node      = int(config_val['VAL']['compute_node']);

        else: # Special for training 

            self.use_gpu       = eval(config_train['SETUP']['use_gpu']);
            self.compute_node  = eval(config_train['SETUP']['compute_node']);
            self.dest_model    = config_train['SETUP']['dest_model'];
            self.runner_id     = int(config_train['SETUP']['runner_id']);
            self.use_run_count = eval(config_train['SETUP']['use_run_count']);



            self.plot_first_comp = eval(config_train['PLOT']['first_comp'])
            self.plot_first_comp = eval(config_train['PLOT']['first_comp'])

        self.plot_first_comp = eval(config_train['PLOT']['first_comp'])
        self.plot_first_comp_general = eval(config_train['PLOT']['first_comp_general'])
        self.plot_second_comp = eval(config_train['PLOT']['second_comp'])
        self.plot_second_comp_general = eval(config_train['PLOT']['second_comp_general'])
        self.plot_f_and_data = eval(config_train['PLOT']['f_and_data'])
        self.plot_f_and_data2 = eval(config_train['PLOT']['f_and_data2'])
        self.plot_data_dist = eval(config_train['PLOT']['data_dist'])
        self.plot_training = eval(config_train['PLOT']['training'])
        self.general_c = float(config_train['PLOT']['c'])


        if config_val is not None:
            config = config_val;
            print('\nUsing config_val\n')
        else:
            config = config_train;
            print('\nUsing config_train\n')

        # Dataset paramters
        self.a             = float(config['DATASET']['a']);
        self.K             = int(config['DATASET']['K']);
        self.eps           = float(config['DATASET']['eps']);
        self.delta         = float(config['DATASET']['delta']);
        self.add_delta     = eval(config['DATASET']['add_delta']);
        self.data_chooser  = config['DATASET']['data_chooser'];
        self.draw_dist     = config['DATASET']['draw_dist'];
        self.data_size     = int(config['DATASET']['data_size']);
        self.train_size    = int(config['DATASET']['train_size']);
        self.val_size      = int(config['DATASET']['val_size']);
        self.test_size     = self.data_size - self.val_size - self.train_size;


        # Training paramters
        self.nbr_epochs   = int(config_train['TRAIN']['nbr_epochs']);
        self.batch_size   = int(config_train['TRAIN']['batch_size']);
        self.shuffle      = eval(config_train['TRAIN']['shuffle']);
        self.optim        = config_train['TRAIN']['optim'];
        self.loss         = config_train['TRAIN']['loss'];
        self.network_str  = config_train['TRAIN']['network'];


        # Gradient decent paramters
        if self.optim.lower() == 'gd':
            start_lr    = float(config_train['TRAIN']['start_lr']); # initial learning rate 
            decay_base  = float(config_train['TRAIN']['decay_base']);
            decay_every = float(config_train['TRAIN']['decay_every']);
            staircase   = eval(config_train['TRAIN']['staircase']);
            self.lr_dict = {'start_lr':    start_lr, 
                       'decay_base':  decay_base, 
                       'decay_every': decay_every,
                       'staircase':   staircase};
        else:
            self.lr_dict = None;

        # Needs to be taken from training setup
        self.prec         = eval(config_train['SETUP']['prec']);
        self.counter_path = config_train['SETUP']['counter_path'];
        if 'TF_CPP_MIN_LOG_LEVEL' in config_train['SETUP'].keys():
            self.tf_log_level   = int(config_train['SETUP']['TF_CPP_MIN_LOG_LEVEL']);
        else:
            self.tf_log_level   = False;
        print_every = int(config_train['TRAIN_SETUP']['print_every']);
        self.print_every = print_every;
        self.save_every = int(eval(config_train['TRAIN_SETUP']['save_every']));
        self.fsize = int(eval(config_train['TRAIN_SETUP']['fsize']));
        self.model_name = config_train['TRAIN_SETUP']['model_name'];
        self.ckpt_dir = config_train['TRAIN_SETUP']['ckpt_dir'];


    def __str__(self):
        if self.config_val is not None:
            val_str = """
#########################################################
###              RUNNER ID: %-5d                     ###
#########################################################

VALIDATION SETUP: 
dest_model        = %s
epoch_nbr         = %d
use_gpu           = %s
compute_node      = %d

""" % (self.runner_id, self.dest_model, self.epoch_nbr, self.use_gpu, 
       self.compute_node);

        train_str = """
DATASET:
a             = %g
K             = %d
eps           = %g
delta         = %g
add_delta     = %s
data_chooser  = %s
draw_dist     = %s
data_size     = %d
train_size    = %d
val_size      = %d
test_size     = %d

SETUP:
use_gpu      = %s
compute_node = %d 
dest_model   = %s
prec         = %s
print_every  = %d
save_every   = %d

TRAIN PARAMTERS:
nbr_epochs   = %d
batch_size   = %d
shuffle      = %s
optim        = %s
loss         = %s
network_str  = %s

""" % ( self.a, self.K, self.eps, self.delta, self.add_delta, 
self.data_chooser, self.draw_dist, self.data_size, 
self.train_size, self.val_size, self.test_size,  self.use_gpu, 
self.compute_node, self.dest_model, self.prec, self.print_every, 
self.save_every, self.nbr_epochs, self.batch_size,  
self.shuffle, self.optim, self.loss, self.network_str); 
        train_header = """
#########################################################
###                TRAINING CONFIG                    ###
#########################################################
"""
            
        if self.config_val is not None:
            return val_str + train_str;
        else:
            return train_header + train_str;











