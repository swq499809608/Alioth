import tensorflow as tf
from .utils import get_from_module

def get(identifier):
    return get_from_module(identifier, globals(), 'loss')

def softmax_categorical_cross_entropy(y_pred, y_true, name="SoftmaxCrossEntropy"):
	with tf.name_scope(name):
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				logits=y_pred, labels=y_true))
		tf.add_to_collection(
			Alioth_Summmaries + '/' + name, 
			tf.summary.scalar(name, loss))
		return loss
