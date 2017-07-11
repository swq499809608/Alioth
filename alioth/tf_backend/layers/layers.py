import tensorflow as tf
import numpy as np 

from ..framework.utils import get_tensor_shape
from ..framework.utils import reshape_tensor
from ..framework.utils import flatten_tensor
from ..framework.utils import autoformat_kernel_2d
from ..framework.utils import autoformat_filter_conv2d
from ..framework.utils import autoformat_padding

from ..framework import initializations as init
from ..framework import activations as act_func

def custom_layer(data_flow, custom_func, **kwargs):
	if 'name' in kwargs:
		name = kwargs['name']
	with tf.name_scope(name):
		tensor = custom_func(data_flow, **kwargs)
	return tensor

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
	with tf.name_scope(name):
		tensor = data_flow
		shape = get_tensor_shape(tensor)
		in_num = int(np.prod(shape[1:]))

		W_init = init.get(weights_init)(shape=[in_num, out_num])
		b_init = init.get(biases_init)(shape=[out_num])
		weights = tf.Variable(W_init)
		biases = tf.Variable(b_init)
		tf.add_to_collection(Alioth_Weights + '/' + name, weights)
		tf.add_to_collection(Alioth_Biases + '/' + name, biases)
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Weights + '/' + name)))+'_weights', weights))
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Biases + '/' + name)))+'_biases', biases))

		if len(shape) > 2:
			tensor = flatten_tensor(tensor)
		tensor = tf.matmul(tensor, weights) + biases
		tensor = act_func.get(activation)(tensor)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)
		
		tensor.weights = weights
		tensor.biases = biases
	return tensor

def dropout(data_flow, keep_prob=0.9, noise_shape=None, train=True, name="Dropout"):
	with tf.name_scope(name):
		tensor = tf.nn.dropout(data_flow, keep_prob, noise_shape=noise_shape, name=name)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)
	return tensor

def conv2d(data_flow, out_channels, filter_size, strides=1, padding='SAME',
		   activation='relu', weights_init='truncated_normal', 
		   biases_init='zeros', name="Conv2D"):
	with tf.name_scope(name):
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
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Weights + '/' + name)))+'_weights', weights))
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Biases + '/' + name)))+'_biases', biases))

		tensor = tf.nn.conv2d(tensor, weights, strides, padding) + biases
		tensor = act_func.get(activation)(tensor)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)

		tensor.weights = weights
		tensor.biases = biases
	return tensor

def atrous_conv2d(data_flow, out_channels, filter_size, rate=1, padding='SAME',
				  activation='relu', weights_init='truncated_normal', 
				  biases_init='zeros', name="AtrousConv2D"):
	with tf.name_scope(name):
		tensor = data_flow
		shape = get_tensor_shape(tensor)
		filter_shape = autoformat_filter_conv2d(filter_size, shape[-1], out_channels)
		padding = autoformat_padding(padding)

		W_init = init.get(weights_init)(shape=filter_shape)
		b_init = init.get(biases_init)(shape=[out_channels])
		weights = tf.Variable(W_init)
		biases = tf.Variable(b_init)
		tf.add_to_collection(Alioth_Weights + '/' + name, weights)
		tf.add_to_collection(Alioth_Biases + '/' + name, biases)
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Weights + '/' + name)))+'_weights', weights))
		tf.add_to_collection(
			Alioth_Summaries + '/train',
			tf.summary.histogram(str(len(tf.get_collection(Alioth_Biases + '/' + name)))+'_biases', biases))

		tensor = tf.nn.atrous_conv2d(tensor, weights, rate, padding) + biases
		tensor = act_func.get(activation)(tensor)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)

		tensor.weights = weights
		tensor.biases = biases
	return tensor

def max_pool_2d(data_flow, kernel_size, strides=None, padding='SAME', name="MaxPooling2D"):
	with tf.name_scope(name):
		tensor = data_flow
		shape = get_tensor_shape(tensor)
		ksize = autoformat_kernel_2d(kernel_size)
		strides = autoformat_kernel_2d(strides) if strides else ksize
		padding = autoformat_padding(padding)
		tensor = tf.nn.max_pool(tensor, ksize, strides, padding=padding)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)
	return tensor

def avg_pool_2d(data_flow, kernel_size, strides=None, padding='SAME', name="AvgPooling2D"):
	with tf.name_scope(name):
		tensor = data_flow
		shape = get_tensor_shape(tensor)
		ksize = autoformat_kernel_2d(kernel_size)
		strides = autoformat_kernel_2d(strides) if strides else ksize
		padding = autoformat_padding(padding)
		tensor = tf.nn.avg_pool(tensor, ksize, strides, padding=padding)
		tf.add_to_collection(Alioth_Tensor + '/' + name, tensor)
	return tensor