import tensorflow as tf
from .utils import get_from_module

def get(identifier):
	if hasattr(identifier, '__call__'):
		return identifier
	else:
		return get_from_module(identifier, globals(), 'initialization')

def zeros(shape=None, dtype=tf.float32):
	if shape is not None:
		return tf.zeros(shape, dtype=dtype)
	else:
		return tf.constant_initializer(value=0.)

def ones(shape=None, dtype=tf.float32):
	if shape is not None:
		return tf.ones(shape=shape, dtype=dtype)
	else:
		return tf.constant_initializer(value=1.)

def random_normal(shape=None, mean=0.0, stddev=0.1, dtype=tf.float32, seed=None):
	if shape is not None:
		return tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
	else:
		return random_normal_initializer(mean=mean, stddev=stddev, seed=seed, dtype=dtype)

def random_uniform(shape=None, minval=0, maxval=None, dtype=tf.float32, seed=None):
	if shape is not None:
		return tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
	else:
		return tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed, dtype=dtype)

def truncated_normal(shape=None, mean=0.0, stddev=0.1, dtype=tf.float32, seed=None):
	if shape is not None:
		return tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
	else:
		return tf.truncated_normal_initializer(mean=mean, stddev=stddev, seed=seed, dtype=dtype)