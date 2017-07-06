
def data_iterator(samples, labels, data_num, iteration_step, chunk_size):
	if len(samples) != len(labels):
		raise Exception("The length of samples and labels must be equal.")
	step = 0
	i = 0
	while i < iteration_step:
		step = (i * chunk_size) % (data_num - chunk_size)
		yield i, samples[step: step + chunk_size], labels[step: step + chunk_size]
		i += 1
