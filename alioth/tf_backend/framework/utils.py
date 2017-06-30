import tensorflow as tf

def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
	if isinstance(identifier, six.string_types):
		res = module_params.get(identifier)
		if not res:
			res = module_params.get(identifier.lower())
			if not res:
				raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
		if instantiate and not kwargs:
			return res()
		elif instantiate and kwargs:
			return res(**kwargs)
		else:
			return res
	return identifier

def get_tensor_shape(tensor):
	if isinstance(tensor, tf.Tensor):
		return tensor.get_shape().as_list()
	elif type(tensor) in [np.array, np.ndarray, list, tuple]:
		return np.shape(tensor)
	else:
		raise Exception("Invalid tensor.")

def reshape_tensor(tensor, shape, name="ReshapeTensor"):
	new = tf.reshape(tensor, shape, name=name)
	return new

def flatten_tensor(tensor, name="FlattenTensor"):
	shape = get_tensor_shape(tensor)
	dims = int(np.prof(shape[1:]))
	new = reshape_tensor(tensor, [-1, dims], name=name)

def one_hot_encoding(target, class_num, on_value=1.0, off_value=0.0, name="OneHotEncoding"):
    with tf.name_scope(name):
        if target.dtype != dtypes.int64:
            target = tf.to_int64(target)
        target = tf.one_hot(target, class_num, on_value=on_value, off_value=off_value)
    tf.add_to_collection(Alioth_Tensor + '/' + name, target)
    return target



def autoformat_kernel_2d(strides):
	if isinstance(strides, int):
		return [1, strides, strides, 1]
	elif isinstance(strides, (tuple, list, tf.TensorShape)):
		if len(strides) == 2:
			return [1, strides[0], strides[1], 1]
		elif len(strides) == 4:
			return [strides[0], strides[1], strides[2], strides[3]]
		else:
			raise Exception("Srides length can only be 2 or 4.")
	else:
		raise Exception("Strides type is not supported.")

def autoformat_filter_conv2d(fsize, in_depth, out_depth):
	if isinstance(fsize,int):
		return [fsize, fsize, in_depth, out_depth]
	elif isinstance(fsize, (tuple, list, tf.TensorShape)):
		if len(fsize) == 2:
			return [fsize[0], fsize[1], in_depth, out_depth]
		else:
			raise Exception("Filter length can only be 2.")
	else:
		raise Exception("Filter type is not supported.")
		
def autoformat_padding(padding):
	if padding == 'same':
		return 'SAME'
	elif padding == 'valid':
		return 'VALID'
	elif padding in ['SAME', 'VALID']:
		return padidng
	else:
		raise Exception("Padding can only be 'same' or 'valid'.")