# What do AI algorithms actually learn? - On False Structures in Deep Learning

This repository contains the code from the paper "What do AI algorithms 
actually learn? - On False Structures in Deep Learning", by L. Thesing, V. Antun
and A. C. Hansen. 

The code for the two experiments can be found in the folders `second_comp` and
`stripe`. In each of the folders the code is organized as follows.

All configurations for training and evaluating the networks can be found in the 
two configuration files `config.ini` and `config_val.ini`, respectively.  
To train a network run `Demo_train_network.py` and to evaluate it run
`Demo_evaluate_net.py`. Whenever you train a network, all data
related to the network will be stored in the folder `models`. The data is stored
in `models/run_x` and `models/plot_x`, where `x` is the "runner_id". i.e. a
number used to identify a specific training setup. 

To recall exactly which parameters a trained neural network has been trained
with, the script `Demo_train_network.py` stores the specific configuration 
file it used and a copy of the script `simple_nn.py` describing the 
architecture in both the run and plot directories.  



