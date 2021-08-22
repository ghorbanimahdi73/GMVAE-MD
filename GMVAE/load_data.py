import numpy as np

class Dataset:
	def __init__(self, data, labels):

		self._x = data
		self._labels = labels
		self.num_data = data.shape[0]
		self.height = data.shape[2]
		self.width = data.shape[1]
		self._idx_batch = 0    # index of batch after initializing the dataset calss

	def next_batch(self, batch_size, shuffle=True):
		start = self._idx_batch
		if start==0:
			if (shuffle):
				idx = np.arange(0, self.num_data)
				np.random.shuffle(idx)
				self._x = self._x[idx]
				self._labels = self._labels[idx]

		# go to the next batch
		if start + batch_size > self.num_data:
			rest_num_data = self.num_data - start
			data_rest_part = self._x[start:self.num_data]
			labels_rest_part = self._labels[start:self.num_data]

			if (shuffle):
				idx0 = np.arange(0, self.num_data)
				np.random.shuffle(idx0)
				self._x = self._x[idx0]
				self._labels = self._labels[idx0]

			start = 0
			# avoid the vase where the number_of_samples != integer times of batch_size
			self._idx_batch = batch_size - rest_num_data
			end = self._idx_batch

			data_new_part = self._x[start:end]
			labels_new_part = self._labels[start:end]

			yield np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self._idx_batch += batch_size
			end = self._idx_batch
		yield self._x[start:end], self._labels[start:end]

	def num_batches(self, batch_size):
		return self.num_data // batch_size


	def random_batch_with_labels(self, batch_size):
		idx = np.arange(0, self.num_data)
		np.random.shuffle(idx)
		return self._x[idx[:batch_size]], self._labels[idx[:batch_size]]

	


