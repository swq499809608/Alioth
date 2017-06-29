import tensorflow as tf
from .utils import get_from_module

def get(identifier):
	if hasattr(identifier, '__call__'):
		return identifier
	else:
		return get_from_module(identifier, globals(), 'regularizer')

def L2(tensor, wd=0.001):
	return tf.multiply(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

def L1(tensor, wd=0.001):
	return tf.multiply(tf.reduce_sum(tf.abs(tensor)), wd, name='L1-Loss')