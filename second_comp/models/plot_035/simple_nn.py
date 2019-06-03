import tensorflow as tf;
import numpy as np;
import keras;

default_dev = '/device:GPU:0';

def generate_training(prediction, label, optim, loss_type, lr_dict=None, batch_size=None, dev_name=default_dev, prec=tf.float32):

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

# Not implemented yet
def ph_test_time(prec=tf.float32, dev_name=default_dev):
    with tf.device(dev_name):
        batch_x      = tf.placeholder(dtype=prec, shape = [None, 2], name='x');
        batch_label  = tf.placeholder(dtype=prec, shape = [None, 1], name='label')    
    return {'batch_x': batch_x, 'batch_label': batch_label};

# Not implemented yet
def test_time_loss(prediction, label, 
                   loss_type='sigmoid_cross_entropy', 
                   dev_name=default_dev):
    
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


def simple_net(batch_x, l_depth, l_width, act, prec, dev_name=default_dev):
    
    hidden = (l_depth)*[None]
    hidden[0] = batch_x;
    with tf.device(dev_name):
        for l in range(l_depth-1):
            hidden[l+1] = tf.keras.layers.Dense(l_width, activation=act, 
                                                dtype=prec)(hidden[l]);
        prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(hidden[-1]);

    return {'pred': prediction};


def constructive_arch(batch_x, K, act, prec, dev_name=default_dev):

    with tf.device(dev_name):
        l1 = tf.keras.layers.Dense(4*K, activation=act, dtype=prec)(batch_x);
        prediction = tf.keras.layers.Dense(1, activation=None,
                                           dtype=prec)(l1);

    return {'pred': prediction};

