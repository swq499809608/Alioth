import tensorflow as tf

def get(identifier):
	if hasattr(identifier, '__call__'):
		return identifier
	else:
		return get_from_module(identifier, globals(), 'evaluate')