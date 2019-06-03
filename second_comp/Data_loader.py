import numpy as np;
import tensorflow as tf;
from scipy.io import loadmat;
from os.path import join;
import sys;

class Data_loader:
    """
Data loader class for the funtion f_a(x) = ceil(a/x_1) % 2, x ∈ ℝ²,
for x_1 in the interval [b, 1], and x_2 ∈ ℝ. For simplicity we take 
`a` to be an integer and set b=a/(K+1) for some integer K > a.

It constructs a dataset where x_1 (the first component) lies in the set

        S_ε = \cup_{k=a}^{K} (a/(k+1)+ε , a/k - ε)

and where the second component x_2, either equals x_2 = 0 for all x_1, or it 
equals x_2 = delta*f_a(x_1).

Attributes
---------
a: Integer, used for f_a
K: Integer, strictly larger than a. We have b = a/(K+1).
eps: epsilon. Have to be less than b**2/(2(a-b)).
delta: Possible extra information added to the second component
b: The lower limit for the interval [b,1] used for the first component
"""
    
    def __init__(self, a, K, eps, delta):
        self.a     = a;
        self.K     = K;
        self.eps   = eps;
        self.delta = delta;
        
        b = a/float(K+1);
        self.b = b;

        if eps > b*b/(2*(a-b)):
            warn_str = """
WARNING: Data_loader: For a: %g, eps: %g and K: %g some of the intervals 
can not be sampled eps stability, and hence they will not be sampled.
            """ % (a, eps, K);
            print(warn_str);
    
    def target_function(self, x):
        """ The function f_a(x) = ceil(a/x_1) % 2 """
        a = self.a;
        return np.ceil(a/x) % 2;

    def load_data_train(self, nbr_of_elements, 
                              add_delta=True, 
                              data_chooser='interval', 
                              dist='uniform'):
        """
Creates random training data, where each sample contain two components x_1, 
and x_2. The samples are redrawn each time one calls the function. 
Hence one can not expect the function to return the same data between two calls.

The first component x_1 will always be chosen so that it lies in 
one for the K-a+1 intervals 
           ( a/(k+1) + epsilon, a/k - epsilon ) for k = a, ..., K.
There are three ways to distribute the samples x_1, within each of these
intervals. The three options can be specified by the `data_chooser` argument, and
may be one of 
- 'interval' - Attempts to place approximately an equal number of samples within
               each of the disjoint intervals. 
- 'random'   - The sample points are drawn at random from a distribution.
- 'half'     - A combination of the above options, i.e. half 'interval' and 
               half 'random'. 
If the 'random' or 'half' options are specified, then all or some of the samples
will be drawn from a distribution specified by the `dist` argument. This may be 
one of 
- 'uniform' - We draw samples from a uniform distribution on [b,1]. 
- 'beta'    - We draw samples from a beta distribution with parameters alpha=1,
              and beta=3. The distribution is shifted to the right by `b`.  

The second component x_2, is chosen in one out of two ways, if the `add_delta`
argument is false, then x_2 = 0. If `add_delta` is true, then 
x_2 = delta*f_a(x_1). 

Arguments
--------
nbr_of_elements : Number of samples.
add_delta: Whether or not we add the delta in the second component.
data_chooser: Type of data we want to draw, one of 'interval', 'random' 
    and 'half'.
dist: If data_chooser is not 'interval', then then use a 'uniform' or 'beta'
    distribution. See above for details.

Returns
-------
x: ndarray, of size [nbr_of_elements, 2], with the chosen samples.
label: ndarray of size [nbr_of_elements, 1] with the value of f_a. 
"""

        if data_chooser.lower() == 'interval':
            x = self._generate_data_in_each_interval(nbr_of_elements);
        elif data_chooser.lower() == 'random':
            x = self._generate_data_random(nbr_of_elements, dist); 
        elif data_chooser.lower() == 'half':
            x = self._generate_data_half_and_half(nbr_of_elements, dist); 
        else:
            print('Data_loader: ERROR, unkown \'data_chooser\', using \'interval\'');
            x = self._generate_data_in_each_interval(nbr_of_elements);

        label = self.target_function(x);
        label = np.reshape(label, [label.shape[0], 1]);
        x = self._add_second_component(x, add_delta=add_delta);
        return x, label;

    def load_data_test(self, nbr_of_elements, add_delta=True, data_chooser='interval', dist='uniform'):
        """ See load_data_train """
        return self.load_data_train(nbr_of_elements, add_delta, data_chooser=data_chooser, dist=dist);

    def load_data_val(self, nbr_of_elements, add_delta=True, data_chooser='interval', dist='uniform'):
        """ See load_data_train """
        return self.load_data_train(nbr_of_elements, add_delta, data_chooser=data_chooser, dist=dist);

    def _add_second_component(self, x, add_delta=True):
        """
For an array x of shape (N,) we add a second component so that the returned
array has shape (N,2). If add_delta=True, the second component will be delta
each time the target function is 1 and 0 otherwise. If add_delta=False, then the
second component is zero. 

Arguments
--------
x: Array of shape (N,)
add_delta (bool): Whether or not to have a non-zero second component whenever
    the target function is 1.

Returns
-------
x: ndarray, of size (N, 2), with the chosen samples.
"""
        out = np.zeros([x.shape[0], 2]);
        out[:,0] = x;
        if add_delta:
            y = self.target_function(x);
            s = self.delta*np.ones(y.shape[0]);
            s *= y;
            out[:,1] = s;
        return out;

    def _generate_data_random(self, nbr_of_elements, dist='uniform'):
        a     = self.a;
        K     = int(self.K);
        eps   = self.eps;
        delta = self.delta;
        b = self.b;

        oversize = int(np.round(1.5*nbr_of_elements));
        missing_nbr_of_elements = nbr_of_elements;
        x = np.array([]);
         
        while missing_nbr_of_elements > 0:
            #print('missing_nbr_of_elements: ', missing_nbr_of_elements);
            if missing_nbr_of_elements > 50:
                oversize = int(np.round(1.5*missing_nbr_of_elements));
            else:
                oversize = int(np.round(10*missing_nbr_of_elements));
            
            if dist.lower() == 'uniform':
                x_new = np.random.uniform(low=b, high=1, size=oversize);
            elif dist.lower() == 'beta':
                x_new = b + np.random.beta(a=1,b=3, size=oversize);
                idx1 = x_new > 1;
                x_new[idx1] = b;
            else:
                print('Data_loader: Error, uknown distribution... using uniform');
                x_new = np.random.uniform(low=b, high=1, size=oversize);
            x_legal = self._remove_illegal_elements(x_new);
            x = np.concatenate((x, x_legal));
            missing_nbr_of_elements = nbr_of_elements-len(x);
        x = x[0:nbr_of_elements];
        np.random.shuffle(x);
        return x;

    def _generate_data_in_each_interval(self, nbr_of_elements):
        a     = self.a;
        K     = int(self.K);
        eps   = self.eps;
        delta = self.delta;
        b     = self.b;

        m = int(np.floor(nbr_of_elements/( K - np.ceil(a) + 1 ) ));
        
        x = np.array([]);
        for k in range(int(np.ceil(a)),K+1):
            lb = (a/(k+1)) + eps;
            ub = (a/k) - eps;
            x_next_interval = np.random.uniform(low=lb, high=ub, size=m);
            x = np.concatenate((x, x_next_interval));
        if nbr_of_elements - len(x) > 0:
            x_random = self._generate_data_random(nbr_of_elements-len(x), dist='beta');
            x = np.concatenate((x,x_random));
        np.random.shuffle(x);
        return x;


    def _generate_data_half_and_half(self, nbr_of_elements, dist='uniform'):

        N = int(np.round(nbr_of_elements/2));
        x = self._generate_data_in_each_interval(N);
        x_random = self._generate_data_random(nbr_of_elements-len(x), dist=dist);
        x = np.concatenate((x,x_random));
        np.random.shuffle(x);
        return x;
    
    def _remove_illegal_elements(self, array):
        """ Removes elements for the array `array` whose values does not lie
        in the intervals (a/(k+1)+epsilon , a/k - epsilon) for k = a, ..., K.
        """
        a     = self.a;
        K     = self.K;
        eps   = self.eps;
        delta = self.delta;

        b = a/float(K+1);
        
        legal_indices = [];
        legal_indices = np.array([],dtype='int64');
        for k in range(int(np.ceil(a)), int(K+1)):
            lb = (a/(k+1)) + eps;
            ub = (a/k) - eps;
            #print('ub: %g, lb: %g' %(ub, lb));
            if (ub < lb):
                print('Warning ub < lb');
            idx = np.where(np.logical_and(array > lb, array < ub));
            legal_indices = np.concatenate((legal_indices, idx[0]));
        return array[legal_indices];    



def create_data_iterator(train_size, shuffle, prec=tf.float32):
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
    data_x = tf.placeholder(dtype=prec, shape = [None, 2], name='data_x');
    data_label  = tf.placeholder(dtype=prec, shape = [None, 1], name='data_label')    

    if shuffle:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat().shuffle(train_size);
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();
    test_dataset  = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it
    val_dataset   = tf.data.Dataset.from_tensor_slices(( data_x, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    batch_x, batch_label  = iter.get_next()

    train_init_op = iter.make_initializer(train_dataset)
    test_init_op  = iter.make_initializer(test_dataset)
    val_init_op   = iter.make_initializer(test_dataset)

    return {'data_x':        data_x, 
            'data_label':    data_label,
            'batch_x':       batch_x,
            'batch_label':   batch_label,
            'train_init_op': train_init_op, 
            'test_init_op':  test_init_op,
            'val_init_op':   val_init_op,
            'ph_batch_size': ph_batch_size};


