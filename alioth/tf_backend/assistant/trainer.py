import tensorflow as tf
from .generator import data_iterator
from .optimizer import optimizer
from ..framework.losses import softmax

def apply_regularization(_lambda):
	regularization = 0.0
	

	