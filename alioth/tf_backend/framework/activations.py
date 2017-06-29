import tensorflow as tf
from .utils import get_from_module

def get(identifier):
	if hasattr(identifier, '__call__'):
		return identifier
	else:
		return get_from_module(identifier, globals(), 'activation')

def linear(tensor):
	return tensor
def tanh(tensor):
	return tf.nn.tanh(tensor)
def sigmoid(tensor):
	return tf.nn.sigmoid(tensor)
def softmatensor(tensor):
	return tf.nn.softmax(tensor)
def softplus(tensor):
	return tf.nn.softplus(tensor)
def softsign(tensor):
	return tf.nn.softsign(tensor)
def relu(tensor):
	return tf.nn.relu(tensor)
def relu6(tensor):
	return tf.nn.relu6(tensor)
def elu(tensor):
	return tf.nn.elu(tensor)
def crelu(tensor):
	return tf.nn.crelu(tensor)