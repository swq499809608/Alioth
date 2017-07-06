import tensorflow as tf
import numpy as np
from tensorflow import confusion_matrix as cfs_mat

def get(identifier):
	if hasattr(identifier, '__call__'):
		return identifier
	else:
		return get_from_module(identifier, globals(), 'evaluate')

def accuracy(labels, predictions, cfsmat_needed=True, per_cent=True, name="Accuracy"):
	with tf.name_scope(name):
		_predictions = np.argmax(predictions, 1)
		_labels = np.argmax(labels, 1)
		cfsmat = cfs_mat(_labels, predictions, name="ConfusionMatrix")
		if per_cent:
			accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
		else:
			accuracy = (np.sum(_predictions == _labels) / predictions.shape[0])
		tf.add_to_collection(
			Alioth_Summmaries + '/' + name, 
			tf.summary.scalar(name, accuracy))
	return accuracy, cfsmat
