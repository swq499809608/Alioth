import tensorflow as tf
import numpy as np 

from ..framework.utils import get_tensor_shape
from ..framework.utils import reshape_tensor
from ..framework.utils import flatten_tensor
from ..framework.utils import autoformat_kernel_2d
from ..framework.utils import autoformat_filter_conv2d
from ..framework.utils import autoformat_padding

from ..framework import initializations as init
from ..framework import activation as act_func

def input_data(shape, dtype=tf.float32, name="InputData"):
	if len(shape) > 1 and shape[0] is not None:
		shape = list(shape)
		shape = [None] + shape
	with tf.name_scope(name):
		placeholder = tf.placeholder(dtype, shape=shape, name="X")
	tf.add_to_collection(Alioth_Inputs, placeholder)
	return placeholder

def fully_connected(data_flow, out_num, activation='relu',
					weights_init='truncated_normal', biases_init='zeros',
					name="FullyConnected"):
	tensor = data_flow
	shape = get_tensor_shape(tensor)
	in_num = int(np.prod(shape[1:]))

	W_init = init.get(weights_init)(shape=[in_num, out_num])
	b_init = init.get(biases_init)(shape=[out_num])
	weights = tf.Variable(W_init)
	biases = tf.Variable(b_init)
	tf.add_to_collection(Alioth_Weights + '/' + name, weights)
	tf.add_to_collection(Alioth_Biases + '/' + name, biases)

	if len(shape) > 2:
		tensor = reshape_tensor(tensor, [-1, in_num], name=None)
	tensor = tf.matmul(tensor, weights) + biases
	tensor = act_func.get(activation)(tensor)
	tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)
	
	tensor.weights = weights
	tensor.biases = biases
	return tensor
	
def conv2d(data_flow, out_channels, filter_size, strides=1, padding='SAME',
		   activation='relu', weights_init='truncated_normal', 
		   biases_init='zeros', name="Conv2D"):
	tensor = data_flow
	shape = get_tensor_shape(tensor)
	filter_shape = autoformat_filter_conv2d(filter_size, shape[-1], out_channels)
	strides = autoformat_kernel_2d(strides)
	padding = autoformat_padding(padding)

	W_init = init.get(weights_init)(shape=filter_shape)
	b_init = init.get(biases_init)(shape=[out_channels])
	weights = tf.Variable(W_init)
	biases = tf.Variable(b_init)
	tf.add_to_collection(Alioth_Weights + '/' + name, weights)
	tf.add_to_collection(Alioth_Biases + '/' + name, biases)

	tensor = tf.nn.conv2d(tensor, weights, strides, padding) + biases
	tensor = act_func(activation)(tensor)
	tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)

	tensor.weights = weights
	tensor.biases = biases
	return tensor

	