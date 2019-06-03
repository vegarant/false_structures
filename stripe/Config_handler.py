import configparser;
import functools;
import os;
import tensorflow as tf;


class Config_handler:
    """
Class to handle the configuration file. 
"""

    def __init__(self, config_train, config_val=None):
        """
The idea of this initialization is to read and set the right config options.

If you are only provide the config_train, 

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

            self.plot_training = eval(config_train['PLOT']['training'])
            self.plot_accuracy = eval(config_train['PLOT']['accuracy'])


        if config_val is not None:
            config = config_val;
            print('\nUsing config_val\n')
        else:
            config = config_train;
            print('\nUsing config_train\n')
        config = config_train;
        # Dataset paramters
        self.n             = int(config['DATASET']['n']);
        self.line_width    = int(config['DATASET']['line_width']);
        self.a             = float(config['DATASET']['a']);
        self.a_low_test    = float(config['DATASET']['a_low_test']);
        self.test_size     = int(config['DATASET']['test_size']);

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
n             = %d
line_width    = %d
a             = %g
a_low_test    = %g
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

""" % ( self.n, self.line_width, self.a, self.a_low_test, self.test_size,
self.use_gpu, self.compute_node, self.dest_model, self.prec, self.print_every,
self.save_every, self.nbr_epochs, self.batch_size,  self.shuffle, self.optim,
self.loss, self.network_str); 
        train_header = """
#########################################################
###                TRAINING CONFIG                    ###
#########################################################
"""
            
        if self.config_val is not None:
            return val_str + train_str;
        else:
            return train_header + train_str;











