#!/usr/bin/env python3

"""
Created by Mahdi Ghorbani
"""

import tensorflow as tf
import numpy as np
from real_conv_GMVAE_graph_mse import Conv_GMVAEgraph
from logger import Logger
from base_model import BaseModel
import sys
from tqdm import tqdm

def lrelu(x, leak=0.2, name='lrelu'):
	""" Leaky recticiter unit
	parameters:
	x:	Tensor
	leak: float, leakage parameter
	name: str, variable scope to use
	returns:
			x: tensor output of nonlinearity
	"""
	with tf.variable_scope(name):
		f1 = 0.5 * (1+leak)
		f2 = 0.5 * (1-leak)
		return f1*x+f2*abs(x)

class ConvGaussianMixtureVAE(BaseModel):
	def __init__(self, input_width, input_height, z_dim=10, K_clusters=10, filters=[16,16],
		         k_sizes=[2,2], paddings=['SAME','SAME'], stride=[1,1], pool_sizes=[2,2], r_nent=1, r_label=1, r_recons=1, temperature=0.1,
		         transfer_fct=lrelu, dense_n=64, learning_rate=0.001, kinit=tf.contrib.layers.xavier_initializer(),
		         bias_init=tf.constant_initializer(0.), batch_size=32, reuse=None, drop_rate=0.3, use_batch_norm=False, phase=True,
		         epochs=10, restore=0, summary_dir='summary', result_dir='result', checkpoint_dir='checkpoint'):

		super().__init__(checkpoint_dir,summary_dir, result_dir)
		self.width = input_width
		self.height = input_height
		self.z_dim = z_dim
		self.K = K_clusters
		self.filters = filters
		self.k_sizes = k_sizes
		self.paddings = paddings
		self.stride = stride
		self.pool_sizes = pool_sizes
		self.transfer_fct = transfer_fct
		self.dense_n = dense_n
		self.learning_rate = learning_rate
		self.kinit = kinit
		self.bias_init = bias_init
		self.batch_size = batch_size
		self.reuse = reuse
		self.drop_rate = drop_rate
		self.use_batch_norm = use_batch_norm
		self.phase = phase
		self.epochs = epochs
		self.restore = restore
		self.summary_dir = summary_dir
		self.result_dir = result_dir
		self.checkpoint_dir = checkpoint_dir
		self.r_nent = r_nent
		self.r_label = r_label
		self.r_recons = r_recons
		self.temperature = temperature
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.cgmvae_graph = Conv_GMVAEgraph(self.width, self.height, self.z_dim, self.K, self.filters,
				                                self.k_sizes, self.paddings, self.stride, self.pool_sizes, self.r_nent, self.r_label, self.r_recons, self.temperature,
				                                self.transfer_fct, self.dense_n, self.learning_rate, self.kinit,
				                                self.bias_init, self.batch_size, self.reuse, self.drop_rate,
				                                self.use_batch_norm, self.phase)

			self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
			print('trainable params: ', self.trainable_count)


	def train_epoch(self, session, logger, train_data):
		loop = tqdm(range(train_data.num_batches(self.batch_size)))

		losses = []
		ent = []
		loss_labeled = []
		loss_recons = []

		for _ in loop:
			batch_x, labels_x = next(train_data.next_batch(self.batch_size))
			loss_e, ent_e, labeled_e, recons_e = self.cgmvae_graph.fit_batch(session, batch_x, self.drop_rate)
			losses.append(loss_e)
			ent.append(ent_e)
			loss_labeled.append(labeled_e)
			loss_recons.append(recons_e)

		losses = np.mean(losses)
		ent = np.mean(ent)
		loss_labeled = np.mean(labeled_e)
		loss_recons = np.mean(recons_e)

		cur_it = self.cgmvae_graph.global_step_tensor.eval(session)

		summary = {'loss':losses, 'ent':ent, 'labeld_loss':loss_labeled, 'recons_loss':loss_recons}
		logger.summarize(cur_it, summaries_dict=summary)

		return losses, ent, loss_labeled, loss_recons

	def valid_epoch(self, session, logger, valid_data):
		loop = tqdm(range(valid_data.num_batches(self.batch_size)))
		losses = []
		ent = []
		loss_labeled = []
		loss_recons = []

		for _ in loop:
			batch_x, labels_x = next(valid_data.next_batch(self.batch_size))
			loss_e, ent_e, labeled_e, recons_e = self.cgmvae_graph.eval_batch(session, batch_x)
			losses.append(loss_e)
			ent.append(ent_e)
			loss_labeled.append(labeled_e)
			loss_recons.append(recons_e)

		losses = np.mean(losses)
		ent = np.mean(ent)
		loss_labeled = np.mean(labeled_e)
		loss_recons = np.mean(recons_e)

		cur_it = self.cgmvae_graph.global_step_tensor.eval(session)

		summary = {'loss':losses, 'ent':ent, 'labeled_loss':loss_labeled, 'recons_loss':loss_recons}
		logger.summarize(cur_it, summaries_dict=summary)

		return losses, ent, loss_labeled, loss_recons

	def train(self, train_data, valid_data):

		with tf.Session(graph=self.graph) as session:
			tf.set_random_seed(1234)

			logger = Logger(session, self.summary_dir)
			saver = tf.train.Saver()

			if (self.restore==1 and self.load(session, saver)):
				num_epochs_trained = self.cgmvae_graph.cur_epoch_tensor.eval(session)
				print('EPOCHs trained: ',num_epochs_trained)
			else:
				print('initializing variables...')
				tf.global_variables_initializer().run()

			if (self.cgmvae_graph.cur_epoch_tensor.eval(session) == self.epochs):
				return

			train_dic = {'loss':[], 'ent':[], 'label':[], 'recons':[]}
			val_dic = {'loss':[], 'ent':[], 'label':[], 'recons':[]}
			for cur_epoch in range(self.cgmvae_graph.cur_epoch_tensor.eval(session), self.epochs+1):


				print('EPOCH: ', cur_epoch)
				self.current_epoch = cur_epoch

				losses_train, ent_train, label_train, recons_train = self.train_epoch(session, logger, train_data)

				train_st = 'TRAIN | Loss: ' + str(losses_train) + ' | Entropy: ' + str(ent_train) + ' | Labeled_loss: ' +str(label_train) + ' | Recons_loss: ' + str(recons_train)

				losses_val, ent_val, label_val, recons_val = self.valid_epoch(session, logger, valid_data)

				valid_st = 'VALID | Loss: ' + str(losses_val) + ' | Entropy: ' + str(ent_val) + ' | Labeld_loss: ' + str(label_val) + ' | Recons_loss: ' + str(recons_val)

				train_dic['loss'].append(losses_train)
				train_dic['ent'].append(ent_train)
				train_dic['label'].append(label_train)
				train_dic['recons'].append(recons_train)

				val_dic['loss'].append(losses_val)
				val_dic['ent'].append(ent_val)
				val_dic['label'].append(label_val)
				val_dic['recons'].append(recons_val)
				print(train_st)
				print(valid_st)
				if (cur_epoch>0 and cur_epoch%10==0):
					self.save(session, saver, self.cgmvae_graph.global_step_tensor.eval(session))

				session.run(self.cgmvae_graph.increment_cur_epoch_tensor)

			self.save(session, saver, self.cgmvae_graph.global_step_tensor.eval(session))

		return train_dic, val_dic





	def generate_embedding(self, data):
		with tf.Session(graph=self.graph) as session:
			saver = tf.train.Saver()
			if (self.load(session, saver)):
				num_epochs_trained = self.cgmvae_graph.cur_epoch_tensor.eval(session)
				print('EPOCHs trained: ', num_epochs_trained)
			else:
				return
			labels = []
			z_recons = []
			y_recons = []
			x_data = []
			for i in range(data.num_batches(batch_size=self.batch_size)):
				x_batch, x_labels = next(data.next_batch(self.batch_size, shuffle=False))
				zm = self.cgmvae_graph.encode_zm(session, x_batch)
				ys = self.cgmvae_graph.encode_y(session, x_batch)
				zm = np.array(zm)
				ys = np.array(ys)
				ys = ys.reshape((self.batch_size, self.K));
				zm = zm.reshape((self.batch_size, self.z_dim));
				#for z_i in range(zs[0].shape[1]):
				#	for y_i in range(ys.shape[1]):
				#		z[:,z_i] += zs[y_i][:,z_i]*ys[:,y_i]   # the embedding [batch_size, self.K]
				z_recons.append(zm);
				labels.append(x_labels); # the label probabilities for each class [batch_size, self.K]
				y_recons.append(ys);
				x_data.append(x_batch);

		return z_recons,y_recons, labels, x_data


	def reconstruct_data(self, data):
		with tf.Session(graph=self.graph) as session:
			saver = tf.train.Saver()
			if (self.load(session, saver)):
				num_batches_trained = self.cgmvae_graph.cur_epoch_tensor.eval(session)
				print('EPOCHs trained: ', num_epochs_trained)
			else:
				return


	def generate_data(self, num_elements=32, category=0):
		""" generate data for a specified category

		args:
			num_elements: (int) number of elements to generate
			category: (int) category from which to generate data

		returns:
			generated data according to num_elements
		"""
		with tf.Session(graph=self.graph) as session:
			saver = tf.train.Saver()
			if (self.load(session, saver)):
				num_batches_trained = self.cgmvae_graph.cur_epoch_tensor.eval(session)
				print('EPOCHs trained: ', num_batches_trained)
			else:
				return

			recons = self.cgmvae_graph.generate_samples(session, category)

		return recons



