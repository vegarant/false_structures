import numpy as np;
import tensorflow as tf;
from os.path import join;
import sys;


class Data_loader:
    """ Data loader class for loading training and test data. 

Images with a horizontal stripe, have a back ground colour of
`-a`, whereas the light stripe has value `1-a`. Images with a vertical
stripe have a background colour value of `a` and a light stripe with 
value `1+a`.

Attributes
----------
n (int): Size of images (n×n)
a (float): Range of colours [-a, 1+a].
line_width: Number of pixels used for the line.
"""



    def __init__(self, n, a,line_width=1):

        self.n     = int(n);
        self.a     = a;
        self.line_width = int(line_width);


    def load_data_train(self, shuffle=True):
        """ Load the training data. Size of this data will be 
`N = 2*(n-line_width+1)`, and contain the unique elements one can create with a
fixed value for `a`. The images have the following colour codes

Horizontal line images:
    - Background colour: -a
    - Stripe colour: 1-a
Vertical line images:
    - Background colour: a
    - Stripe colour: 1+a

If shuffle=False, then these elements will occur in an order where all the
horizontal lines precede the vertical lines.

Arguments
---------
suffle (bool): Whether or not to shuffle the images.


Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""

        data, label = self._generate_all_combinations_of_stripe_images(shuffle=shuffle);

        return data, label;


    def load_data_test(self, size, a_low, a_high=None):
        """ Load test data. 
Loads test data set of size `size`. The values of `a` will be drawn from 
a uniform distribution on the interval [a_low, a_high]. The images will have 
the following colour codes. 

Horizontal line images:
    - Background colour: -a
    - Stripe colour: 1-a
Vertical line images:
    - Background colour: a
    - Stripe colour: 1+a

Arguments
---------
size (int): Size of the test set
a_low (float): lower bound on the `a` values.
a_high: If a_high=None, then a_high will equal the class attribute `a`. 
    Otherwise a_high should be a float strictly larger than a_low.

Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""

        if a_high is None:
            a_high = self.a;

        data, label = self._generate_test_set(size, a_low, a_high, flip_structure=False);

        return data, label;


    def load_data_test_false(self, size, a_low, a_high=None):
        """ Load false test data. 
Loads test data set of size `size`. The values of `a` will be drawn from 
a uniform distribution on the interval [a_low, a_high]. The images will have 
the following colour codes, which are the opposite of the colour codes used 
for `load_data_test`.

Horizontal line images:
    - Background colour: a
    - Stripe colour: 1+a
Vertical line images:
    - Background colour: -a
    - Stripe colour: 1-a

Arguments
---------
size (int): Size of the test set
a_low (float): lower bound on the `a` values.
a_high: If a_high=None, then a_high will equal the class attribute `a`. 
    Otherwise a_high should be a float strictly larger than a_low.

Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""
        if a_high is None:
            a_high = self.a;

        data, label = self._generate_test_set(size, a_low, a_high, 
                                              flip_structure=True);

        return data, label;


    def load_data_val(self, size, a_low, a_high=None):
        """ Load validation data. 
Loads test data set of size `size`. The values of `a` will be drawn from 
a uniform distribution on the interval [a_low, a_high]. The images will have 
the following colour codes. 

Horizontal line images:
    - Background colour: -a
    - Stripe colour: 1-a
Vertical line images:
    - Background colour: a
    - Stripe colour: 1+a

Arguments
---------
size (int): Size of the test set
a_low (float): lower bound on the `a` values.
a_high: If a_high=None, then a_high will equal the class attribute `a`. 
    Otherwise a_high should be a float strictly larger than a_low.

Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""
        data, label = self._generate_test_set(size, a_low, a_high, 
                                              flip_structure=True);

        
        return data, label;

    def _generate_all_combinations_of_stripe_images(self, shuffle=True):

        a = self.a;
        n = self.n;
        line_width = self.line_width;
        lidx = np.array(list(range(line_width)));

        nbr_of_perm = n-line_width+1;
        data_hor = -a*np.ones([nbr_of_perm, n, n]);
        data_ver =  a*np.ones([nbr_of_perm, n, n]);
        label_hor = np.ones([nbr_of_perm, 1]);
        label_ver = np.zeros([nbr_of_perm, 1]);

        for i in range(nbr_of_perm):
            data_hor[i, i+lidx, :] += 1; 
            data_ver[i, i+lidx, :] += 1; 

        data = np.concatenate((data_hor, data_ver), axis=0);
        label = np.concatenate((label_hor, label_ver), axis=0);

        if shuffle:
            idx = np.arange(2*nbr_of_perm);
            np.random.shuffle(idx);
            data = data[idx];
            label = label[idx];

        p = n*line_width;
        for i in range(2*nbr_of_perm):
            s = np.sum(data[i,:,:]);
            #print('s < p: %6s, s: %g, p: %g, label[i]: %d' % (s<p,s,p,label[i]))
            # s < p implies horizontal, label for horizontal is 1
            # s > p implies vertical, label for vertical is 0
            if s < p and label[i] < 0.5:
                print(i, 'Image wrongly classified as vertical')
            if s > p and label[i] > 0.5:
                print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        return data, label;

    def _generate_test_set(self, size, a_low, a_high, flip_structure=False):

        a = self.a;
        n = self.n;
        line_width = self.line_width;

        stripe_idx = np.random.randint(low=0, high=n+1-line_width, size=size);
        a_values = np.random.uniform(low=a_low, high=a_high, size=size);
        is_horizontal = np.random.choice(a=[True, False], size=size, p=[0.5, 0.5])
        lidx = np.array(list(range(line_width)));

        data = np.zeros([size, n, n]);
        label = np.zeros([size, 1]);
        for i in range(size):

            if is_horizontal[i]:
                sign = -1;
            else:
                sign = 1;

            if flip_structure:
                sign = -sign;

            data[i, :, :] = sign*a_values[i]; 

            if is_horizontal[i]:
                data[i, stripe_idx[i]+lidx, :] += 1; 
                label[i] = 1;
            else:
                data[i, :, stripe_idx[i]+lidx] += 1; 
                label[i] = 0;

        p = self.line_width*self.n;
        if flip_structure:
            for i in range(size):
                s = np.sum(data[i,:,:]);
                if s > p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s < p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        else:
            for i in range(size):
                s = np.sum(data[i,:,:]);
                #print('s < p: %6s, s: %g, p: %g, label[i]: %d' % (s<p,s,p,label[i]))
                # s < p implies horizontal, label for horizontal is 1
                # s > p implies vertical, label for vertical is 0
                if s < p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s > p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        return data, label;


def create_data_iterator(train_size, n, shuffle, prec=tf.float32):
    """Creates an iterator which can feed the data efficiently to Tensorflow. 

Arguments
---------
train_size: Size of the training set.
shuffle (bool): Whether or not to shuffle the training set in each.
prec: Tensorflow precision

Returns
-------
iterator_elements (dict): A dictionary with the keys
    'data_x': tf.placeholder for the entire data set.
    'data_label:' tf.placeholder for the labels to the entire data set
    'batch_x': tf.Tensor for a batch of data.
    'batch_label': tf.Tensor for a batch of labels.
    'train_init_op': Iterator for the training set.
    'test_init_op': Iterator for the training set.
    'val_init_op': Iterator for the training set.
    'ph_batch_size': tf.placeholder for the batch size used for the 
        training, test and validation set.
"""

    ph_batch_size = tf.placeholder(tf.int64)
    data_x = tf.placeholder(dtype=prec, shape = [None, n,n], name='data_x');
    data_label  = tf.placeholder(dtype=prec, shape = [None, 1], name='data_label')    

    if shuffle:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat().shuffle(train_size);
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();
    test_dataset  = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it
    val_dataset   = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    #with tf.device(dev_name): 
    batch_x, batch_label  = iter.get_next()

    train_init_op = iter.make_initializer(train_dataset)
    test_init_op  = iter.make_initializer(test_dataset)
    val_init_op   = iter.make_initializer(val_dataset)

    return {'data_x':        data_x, 
            'data_label':    data_label,
            'batch_x':       batch_x,
            'batch_label':   batch_label,
            'train_init_op': train_init_op, 
            'test_init_op':  test_init_op,
            'val_init_op':   val_init_op,
            'ph_batch_size': ph_batch_size};
























