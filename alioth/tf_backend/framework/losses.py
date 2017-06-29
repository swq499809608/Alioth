import tensorflow as tf
from .utils import get_from_module

def get(identifier):
    return get_from_module(identifier, globals(), 'loss')

def softmax_categorical_cross_entropy(y_pred, y_true):
	with tf.name_scope("SoftmaxCrossEntropy"):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=y_pred, labels=y_true))
