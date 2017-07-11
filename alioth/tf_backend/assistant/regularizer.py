import tensorflow as tf

def apply_regularization(lambda_value, layer_name):
	regularization = 0.0
	weights = tf.get_collection(Alioth_Weights + '/' + layer_name)
	bises = tf.get_collection(Alioth_Weights + '/' + layer_name)
	regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
	return lambda_value * regularization
