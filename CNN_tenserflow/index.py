import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("MNIST/",one_hot=True)

learning_rate = 0.01
epochs = 20
batch_size = 256
num_batches = int(mnist.train.num_examples/batch_size)
in_h = 28
in_w = 28
n_classes = 10
display_step = 1
f_size = 5
dep_in = 1
dep_01 = 64
dep_02 = 128



