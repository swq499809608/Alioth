import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import alioth
from alioth.tf_backend.layers import layers
from alioth.tf_backend.framework.evaluate import accuracy
from alioth.tf_backend.framework.losses import softmax_categorical_cross_entropy
from alioth.tf_backend.assistant.optimizer import optimizer

x = layers.input_data([None, 784])
y_ = layers.input_data([None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

net = layers.conv2d(x_image, 32, [5, 5, 1, 32])
net = layers.max_pool_2d(net, [1, 2, 2, 1])
net = layers.conv2d(net, 64, [5, 5, 32, 64])
net = layers.fully_connected(net, [1024])
net = layers.dropout(net)
net = layers.fully_connected(net, [10])

loss = softmax_categorical_cross_entropy(net, y_)
optimizer = optimizer(loss, 1e-4)
for i in range(20000):
	sess = tf.InteractiveSession()
	sess.run(tf.globale_variables_initializer())
	batch = mnist.train.next_batch(50)
	l, pred = sess.run(
		[optimizer, loss, net],
		feed_dict={x: batch[0], y: batch[1]}
	)
	accuracy, _ = accuracy(pred, batch[1])
	if i % 50 == 0:
		print('第 %d 轮训练时的损失值: %f' % (i, loss))
		print('此轮训练的准确度: %.1f%%' % accuracy)
sess.close()