import tensorflow as tf
from ..framework.utils import get_from_module

def exponential_decay(learning_rate, global_step=tf.Variable(0), 
					  decay_steps=100, decay_rate=0.99, staircase=False):
	learning_rate = tf.train.exponential_decay(
		learning_rate = learning_rate,
		global_step = global_step,
		decay_steps = decay_steps,
		decay_rate = decay_rate,
		staircase = staircase
	)

def optimizer(loss, learning_rate, optimizeMethod='adam', 
			 global_step=tf.Variable(0), decay_steps=100, 
			 decay_rate=0.99, staircase=False):
	exponential_decay(
		learning_rate,
		global_step = global_step,
		decay_steps = decay_steps,
		decay_rate = decay_rate,
		staircase = staircase
	)
	optimizer = None
	with tf.name_scope('optimizer'):
		if(optimizeMethod == 'gradient'):
			optimizer = tf.train \
			.GradientDescentOptimizer(learning_rate)\
			.minimize(loss)
		elif(optimizeMethod=='momentum'):
			optimizer = tf.train \
				.MomentumOptimizer(learning_rate, 0.5) \
				.minimize(loss)
		elif(optimizeMethod=='adam'):
			optimizer = tf.train \
				.AdamOptimizer(learning_rate) \
				.minimize(loss)
		tf.add_to_collection('Alioth_Ops' + '/Optimizer', optimizer)
	return optimizer
