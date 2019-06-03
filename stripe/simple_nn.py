"""

This file contains the networks architectures and set up some of the required
tools for training and evaluation of the networks. 
"""
import tensorflow as tf;
import numpy as np;
import keras;

default_dev = '/device:GPU:0';

def generate_training(prediction, label, optim, loss_type, lr_dict=None, batch_size=None, dev_name=default_dev, prec=tf.float32):
    """Set up the type of training one wants to preform, using the specified
loss function and optimization procedure.

Arguments
---------
prediction (tf.Tensor): Output of the network (before any sigmoid). 
label (tf.placeholder): Correct label 
optim (str): Name of the optimizer to use. Supported types are
    'gd': gradient descent
    'adam': ADAM optimizer.
lr_dict (dict): Dictionary with values to use for the gradient descent 
    optimizer.
batch_size: Not in use
dev_name (str): Name of device to place the tensors.
prec: Tensorflow precision.

Returns
-------
placeholder_dict (dict): A dictionary with the keys
    'optimizer': Tensorflow optimizer 
    'loss' (tf.Tensor): Loss of the evaluated data.
"""

    if lr_dict is not None and optim.lower() == 'gd':
        start_lr    = lr_dict['start_lr'];
        decay_every = lr_dict['decay_every'];
        decay_base  = lr_dict['decay_base'];
        staircase   = lr_dict['staircase'];


    global_step = tf.Variable(initial_value=0, trainable=False);
    with tf.device(dev_name):

        if loss_type.lower() == 'mean_squared_error':
            loss = tf.losses.mean_squared_error(label, prediction);    
        elif loss_type.lower() == 'sigmoid_cross_entropy':
            loss = tf.losses.sigmoid_cross_entropy(label, prediction);
        else:
            print('Did not recognize loss type: %s, using mean squared' % loss_type);
            loss = tf.reduce_sum(tf.pow(prediction-label, 2));   

        if optim.lower() == 'gd':

            learning_rate = tf.train.exponential_decay(
                                                 learning_rate = start_lr,
                                                 global_step   = global_step,
                                                 decay_steps   = decay_every,
                                                 decay_rate    = decay_base,
                                                 staircase     = staircase);

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                                         loss, 
                                                         global_step=global_step);

        elif optim.lower() == 'adam': 
            optimizer = tf.train.AdamOptimizer().minimize(loss); 
        
        
    
    return {'optimizer': optimizer,
            'loss': loss};

def ph_test_time(prec, n, dev_name=default_dev):
    """ Creates placeholders for input and labels test time evaluation 
    
Arguments
---------
prec: Tensorflow precision.
dev_name: Name of device to place the Tensors.

Returns
-------
placeholder_dict (dict): A dictionary with the keys
    'batch_x': tf.Tensor for a batch of data.
    'batch_label': tf.Tensor for a batch of labels.
    """
    with tf.device(dev_name):
        batch_x      = tf.placeholder(dtype=prec, shape = [None, n,n], name='x');
        batch_label  = tf.placeholder(dtype=prec, shape = [None, 1], name='label')    
    return {'batch_x': batch_x, 'batch_label': batch_label};


def test_time_loss(prediction, label, 
                   loss_type='sigmoid_cross_entropy', 
                   dev_name=default_dev):
    """ Specify which loss function to use.

Arguments
---------
prediction (tf.Tensor): Output of the network
label (tf.Tensor): Label for the network
loss_type: type of loss function. One of 'sigmoid_cross_entropy'
    and 'mean_squared_error'.
dev_name (str): Name of device for the new Tensors.

Returns
-------
placeholder_dict (dict): A dictionary with the keys
    'accuracy' (tf.Tensor): Accuracy of the network.
    'batch_loss' (tf.Tensor): Loss of the evaluated data.
"""
    
    with tf.device(dev_name):
        if loss_type.lower() == 'sigmoid_cross_entropy':
            pred_binary = tf.round(tf.nn.sigmoid(prediction));
            tmp = tf.cast(tf.equal(pred_binary, label), tf.float32)
            accuracy = tf.math.reduce_mean(tmp);
            batch_loss = tf.math.reduce_sum(tmp);
        elif loss_type.lower() == 'mean_squared_error':
            pred_binary = tf.round(prediction);
            tmp = tf.cast(tf.equal(pred_binary, label), tf.float32)
            accuracy = tf.math.reduce_mean(tmp);
            batch_loss = tf.math.reduce_sum(tmp);
            
    return {'accuracy': accuracy, 'batch_loss': batch_loss};


def conv_net(batch_x, filter_size, act, prec, dev_name=default_dev):
    with tf.device(dev_name):
        input1 = tf.expand_dims(batch_x, axis=-1);
        l1 = tf.keras.layers.Conv2D(filters=filter_size,
                               kernel_size=3,
                               strides=(1,1),
                               padding='same',
                               activation=act,
                               dtype=prec)(input1);
        
        l2 = tf.keras.layers.Conv2D(filters=filter_size,
                               kernel_size=3,
                               strides=(2,2),
                               padding='same',
                               activation=act,
                               dtype=prec)(l1);

        l3 = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=3,
                               strides=(2,2),
                               padding='same',
                               activation=act,
                               dtype=prec)(l2);
        l3_flatt = tf.keras.layers.Flatten(dtype=prec)(l3)
        prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l3_flatt);

    return {'pred': prediction};

def conv2_net(batch_x, filter_size, act, prec, dev_name=default_dev):
    with tf.device(dev_name):
        input1 = tf.expand_dims(batch_x, axis=-1);
        l1 = tf.keras.layers.Conv2D(filters=filter_size,
                               kernel_size=3,
                               strides=(1,1),
                               padding='same',
                               activation=act,
                               dtype=prec)(input1);
        
        l2 = tf.keras.layers.Conv2D(filters=filter_size,
                               kernel_size=3,
                               strides=(2,2),
                               padding='same',
                               activation=act,
                               dtype=prec)(l1);

        l3 = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=3,
                               strides=(2,2),
                               padding='same',
                               activation=act,
                               dtype=prec)(l2);
        l3_flatt = tf.keras.layers.Flatten(dtype=prec)(l3);
        l4 = tf.keras.layers.Dense(10,dtype=prec)(l3_flatt);
        prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l4);

    return {'pred': prediction};


def conv3_net(batch_x, act, prec, dev_name=default_dev):
    input1 = tf.expand_dims(batch_x, axis=-1);
    l1 = tf.keras.layers.Conv2D(filters=24,
                           kernel_size=5,
                           strides=(1,1),
                           padding='same',
                           activation=act,
                           dtype=prec)(input1);
    
    l2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), 
                                   padding='same', 
                                   strides=None)(l1)

    
    l3 = tf.keras.layers.Conv2D(filters=48,
                           kernel_size=5,
                           strides=(1,1),
                           padding='same',
                           activation=act,
                           dtype=prec)(l2);
    
    l4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), 
                                   padding='same', 
                                   strides=None)(l3)

    l4_flatt = tf.keras.layers.Flatten(dtype=prec)(l4);
    
    l5 = tf.keras.layers.Dense(10,dtype=prec)(l4_flatt);
    prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l5);

    return {'pred': prediction};

def conv4_net(batch_x, act, prec, dev_name=default_dev):
    input1 = tf.expand_dims(batch_x, axis=-1);
    l1 = tf.keras.layers.Conv2D(filters=24,
                           kernel_size=5,
                           strides=(1,1),
                           padding='same',
                           activation=act,
                           dtype=prec)(input1);
    
    l2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), 
                                   padding='same', 
                                   strides=None)(l1)

    
    l3 = tf.keras.layers.Conv2D(filters=48,
                           kernel_size=5,
                           strides=(1,1),
                           padding='same',
                           activation=act,
                           dtype=prec)(l2);
    
    l4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), 
                                   padding='same', 
                                   strides=None)(l3)
    
    l5 = tf.keras.layers.Conv2D(filters=64,
                           kernel_size=5,
                           strides=(1,1),
                           padding='same',
                           activation=act,
                           dtype=prec)(l4);
    
    l6 = tf.keras.layers.MaxPool2D(pool_size=(2,2), 
                                   padding='same', 
                                   strides=None)(l5)

    l6_flatt = tf.keras.layers.Flatten(dtype=prec)(l6);
    
    l7 = tf.keras.layers.Dense(256,dtype=prec)(l6_flatt);
    prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l7);

    return {'pred': prediction};

def sum_net(batch_x, prec, dev_name=default_dev):

    l_flatt = tf.keras.layers.Flatten(dtype=prec)(batch_x);

    prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l_flatt);

    return {'pred': prediction};
