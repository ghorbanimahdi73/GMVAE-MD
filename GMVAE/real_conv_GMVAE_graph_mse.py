import numpy as np
import tensorflow as tf
from base_graph import BaseGraph
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

class Conv_GMVAEgraph(BaseGraph):
	def __init__(self, input_width, input_height, z_dim, K_clusters, filters=[32,32], k_sizes=[4,4], paddings=['SAME','SAME'],
		         stride=[1,1], pool_sizes=[1,1], r_nent=1, r_label=1, r_recons=1, temperature=0.1, transfer_fct=tf.nn.relu, dense_n=64, learning_rate=0.001, kinit=tf.contrib.layers.xavier_initializer(),
		         bias_init=tf.constant_initializer(0.), batch_size=32, reuse=None, drop_rate=0.3, use_batch_norm=False, phase=True):

		super().__init__(learning_rate)

		self.width = input_width
		self.height = input_height
		self.filters = filters
		self.k_sizes = k_sizes
		self.paddings = paddings
		self.stride = stride
		self.K = K_clusters
		self.z_dim = z_dim
		self.x_flat_dim = self.width*self.height
		self.pool_sizes = pool_sizes
		self.transfer_fct = transfer_fct
		self.kinit = kinit
		self.bias_init = bias_init
		self.reuse = reuse
		self.use_batch_norm = use_batch_norm
		self.phase = phase
		self.dense_n = dense_n
		self.batch_size = batch_size
		self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, 1], name='x_batch')
		self.x_batch_flat = tf.reshape(self.x_batch, [-1, self.x_flat_dim])
		self.drop_rate = tf.placeholder(tf.float32, shape=(), name='drop_rate')
		self.x_binarized = tf.cast(tf.greater(self.x_batch, tf.random_uniform(tf.shape(self.x_batch), 0, 1)), tf.float32)
		self.temperature= temperature
		self.r_nent = r_nent
		self.r_label = r_label
		self.r_recons = r_recons
		self.hard_gumbel = False
		self.build_network()
		self.build_loss_optimizer()

	def Qy_graph(self, x):
		"""
		Q(y|x) computation graph
		inputs: x (data)
		returns Qy_x or the probability of the datapoint to be in each class (cluster)
		"""
		print('Defining Q(y|x)')
		reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qy_x')) > 0
		with tf.variable_scope('Qy_x'):


			Qy_x = tf.layers.conv2d(inputs=x,
				                    filters=self.filters[0],
				                    kernel_size=self.k_sizes[0],
				                    strides=self.stride[0],
				                    padding=self.paddings[0],
				                    name='qy_x_1',
				                    activation=self.transfer_fct,
				                    kernel_initializer=self.kinit,
				                    bias_initializer=self.bias_init,
				                    reuse=reuse)


			if self.use_batch_norm:
				Qy_x = batch_norm(Qy_x, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_'+str(j+1))

			Qy_x = tf.layers.max_pooling2d(inputs=Qy_x,
				                           pool_size=self.pool_sizes[0],
				                           strides=self.pool_sizes[0],
				                           name='pool_1')


			Qy_x = tf.layers.dropout(Qy_x, rate=self.drop_rate, name='H1_dropout')

			for j in range(1,len(self.filters)):

				Qy_x = tf.layers.conv2d(inputs=Qy_x,
					                    filters=self.filters[j],
					                    kernel_size=self.k_sizes[j],
					                    strides=self.stride[j],
					                    name='conv_'+str(j+1),
					                    activation=self.transfer_fct,
					                    kernel_initializer=self.kinit,
					                    bias_initializer=self.bias_init,
					                    padding=self.paddings[j],
					                    reuse=reuse)

				if self.use_batch_norm:
					Qy_x = batch_norm(Qy_x, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_'+str(j+1))

				Qy_x = tf.layers.max_pooling2d(inputs=Qy_x,
					                           pool_size=self.pool_sizes[j],
					                           strides=self.pool_sizes[j],
					                           name='pool_'+str(j+1))



				Qy_x = tf.layers.dropout(Qy_x, rate=self.drop_rate, name='H'+str(1+j)+'_dropout')

			#Qy_x = tf.layers.max_pooling2d(inputs=Qy_x,
			#	                           pool_size=self.pool_sizes,
			#	                           strides=self.pool_sizes,
			#	                           name='pool_'+str(j+1))



			print('Qy_x conv shape:', Qy_x.get_shape().as_list())
			self.Qy_x_shape = Qy_x.get_shape().as_list()

			output = tf.contrib.layers.flatten(Qy_x)
			self.flattened_Qy_x_conv_shape = output.get_shape().as_list()[-1]

			print('flattened_Qy_x shape: ', self.flattened_Qy_x_conv_shape)
			qy_logit = tf.layers.dense(inputs=output,
				                       units=self.K,
				                       activation=None,
				                       reuse=reuse,
				                       name='qy_logit')


			categorical = self.gumbel_softmax(qy_logit, self.temperature, self.hard_gumbel)

			# sample from the gumbel softmax distribution
			qy = tf.nn.softmax(qy_logit, name='prob') # probability of each cluster

			#log_qy = tf.nn.softmax(qy, name='log_probability')

		return qy_logit, qy, categorical

	def Qz_graph(self, x, y):
		"""
		Q(z|x,y) computational graph
		Encoder model to learn mean and variance of each gaussian
		returns: z, z_mean, z_var
		"""
		print('Defining Q(z|x,y)')
		reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qz_xy')) > 0
		print('reuse:', reuse)
		with tf.variable_scope('Qz_xy'):

			print('shape of y: ',y.get_shape())
			print('shape of x: ',x.get_shape())
			#xy = tf.concat((x, y),1, name='xy_concat')

			net = tf.layers.conv2d(inputs=x,
				                   filters=self.filters[0],
				                   kernel_size=self.k_sizes[0],
				                   strides=self.stride[0],
				                   padding=self.paddings[0],
				                   name='QzConv_1',
				                   activation=self.transfer_fct,
				                   kernel_initializer=self.kinit,
				                   bias_initializer=self.bias_init,
				                   reuse=reuse)

			if self.use_batch_norm:
				net = batch_norm(net, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_net')

			net = tf.layers.max_pooling2d(inputs=net,
				                          pool_size=self.pool_sizes[0],
				                          strides=self.pool_sizes[0],
				                          name='Qzpool_1')

			net = tf.layers.dropout(net, rate=self.drop_rate, name='conv1_dropout')

			for j in range(1, len(self.filters)):
				net = tf.layers.conv2d(inputs=net,
					                   filters=self.filters[j],
					                   kernel_size=self.k_sizes[j],
					                   strides=self.stride[j],
					                   padding=self.paddings[j],
					                   name='QzConv_'+str(j+1),
					                   activation=self.transfer_fct,
					                   kernel_initializer=self.kinit,
					                   bias_initializer=self.bias_init,
					                   reuse=reuse)
				if self.use_batch_norm:
					net = batch_norm(net, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_net_'+str(j+1))

				net = tf.layers.max_pooling2d(inputs=net,
					                          pool_size=self.pool_sizes[j],
					                          strides=self.pool_sizes[j],
					                          name='Qzpool_'+str(j+1))

				net = tf.layers.dropout(net, rate=self.drop_rate, name='Qzconv'+str(j+1)+'_dropout')


			# net = tf.layers.max_pooling2d(inputs=net,
			# 	                          pool_size=self.pool_sizes,
			# 	                          strides=self.pool_sizes,
			# 	                          name='Qzpool_'+str(j+1))


			print('Qz_x conv shape: ', net.get_shape())
			self.Qz_x_conv_shape = net.get_shape().as_list()

			output = tf.contrib.layers.flatten(net)
			self.flattened_Qz_x_conv_shape = output.get_shape().as_list()[-1]
			print('flattened:', self.flattened_Qz_x_conv_shape)

			xy = tf.concat((output, y), 1, name='xy_concat')

			print('shape of xy:', xy.get_shape())

			qz = tf.layers.dense(inputs=xy,
				                 units=self.dense_n,
				                 activation=self.transfer_fct,
				                 reuse=reuse,
				                 kernel_initializer=self.kinit,
				                 bias_initializer=self.bias_init,
				                 name='qz_1')


			if self.use_batch_norm:
				qz = batch_norm(qz, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_xy')

			qz = tf.layers.dropout(qz, rate=self.drop_rate, name='xy_1_dropout')

			for i in range(1, len(self.filters)):
				qz = tf.layers.dense(inputs=qz,
					                 units=self.dense_n,
					                 activation=self.transfer_fct,
					                 reuse=reuse,
					                 kernel_initializer=self.kinit,
					                 bias_initializer=self.bias_init,
					                 name='qz_'+str(i+1)+'graph')
				
				if self.use_batch_norm:
					qz = batch_norm(qz, center=True, scale=True, is_training=self.phase, reuse=reuse, name='batch_norm_xy'+str(i+1))

				qz = tf.layers.dropout(qz, rate=self.drop_rate, name='xy_'+str(i+1)+'dropout')

			zm = tf.layers.dense(inputs=qz,
				                 units=self.z_dim,
				                 activation=None,
				                 reuse=reuse,
				                 name='qz_zm')

			zv = tf.layers.dense(inputs=qz,
				                 units=self.z_dim,
				                 activation=tf.nn.softplus,
				                 reuse=reuse,
				                 name='qz_zv')+1e-5

		print('Reparameterization trick...')
		z_xy = tf.random_normal((self.batch_size, self.z_dim), zm, tf.sqrt(zv), dtype=tf.float32, name='z_xy')
		#z_xy = tf.add(zm, tf.multiply(tf.sqrt(zv), eps))
		print('shape of z:', z_xy.get_shape())
		return z_xy, zm, zv

	def Pz_graph(self, y):
		"""
		P(z|y) computational graph
		"""
		reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pz_y')) > 0
		with tf.variable_scope('pz_y'):


			zm = tf.layers.dense(inputs=y,
				                 units=self.z_dim,
				                 activation=None,
				                 reuse=reuse,
				                 name='pz_zm')

			zv = tf.layers.dense(inputs=y,
				                 units=self.z_dim,
				                 activation=tf.nn.softplus,
				                 reuse=reuse,
				                 name='pz_zv') + 1e-5

		#self.z_y_mean = zm
		#self.z_y_var = zv
		#print('Reparameterization trick...')
		#eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
		#self.z_y = tf.add(self.z_y_mean, tf.multiply(tf.sqrt(self.z_y_var), eps))

		return zm, zv

	def Px_graph(self, z):
		"""
		P(x|z) computation graph.
		"""
		reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0
		with tf.variable_scope('px'):
			px = tf.layers.dense(inputs=z,
				                 units=self.dense_n,
				                 activation=self.transfer_fct,
				                 kernel_initializer=self.kinit,
				                 bias_initializer=self.bias_init,
				                 reuse=reuse,
				                 name='px_1')

			px = tf.layers.dropout(px, rate=self.drop_rate, name='px_1_dropout')

			px = tf.layers.dense(inputs=px,
				                 units=self.flattened_Qz_x_conv_shape,
				                 activation=self.transfer_fct,
				                 kernel_initializer=self.kinit,
				                 bias_initializer=self.bias_init,
				                 name='px_dense_2',
				                 reuse=reuse)


			print('decoder_dense shape: ', px.get_shape())

			deconv_shape = tf.reshape(px, self.Qz_x_conv_shape)


			deconv_reshape = tf.layers.conv2d_transpose(inputs=deconv_shape,
				                                        filters=self.filters[-1],
				                                        kernel_size=self.k_sizes[-1],
				                                        strides=self.pool_sizes[-1],
				                                        padding=self.paddings[-1],
				                                        activation=self.transfer_fct,
				                                        name='deconv_transpose',
				                                        reuse=reuse)

			print('deconv_shape: ', deconv_reshape.get_shape())

			for j in reversed(range(1,len(self.filters)-1)):

				deconv_reshape = tf.layers.conv2d_transpose(inputs=deconv_reshape,
					                                        filters=self.filters[j],
					                                        kernel_size=self.k_sizes[j],
					                                        strides=self.pool_sizes[j],
					                                        padding=self.paddings[j],
					                                        activation=self.transfer_fct,
					                                        name='deconv_'+str(j+1),
					                                        reuse=reuse)

				if self.use_batch_norm:
					deconv_reshape = batch_norm(deconv_reshape, center=True, scale=True, is_training=self.phase, name='dense_bn'+str(i))

				deconv_reshape = tf.layers.dropout(deconv_reshape, rate=self.drop_rate, name='dense'+str(j+1)+'_dropout')

				print('deconv_shape: ', deconv_reshape.get_shape())


			px_logit = tf.layers.conv2d(inputs=deconv_reshape,
				                                  filters=1,
				                                  kernel_size=self.k_sizes[0],
				                                  strides=self.pool_sizes[0],
				                                  padding=self.paddings[0],
				                                  activation=None,
				                                  name='px_final',
				                                  reuse=reuse)

			print('px_logit shape: ', px_logit.get_shape())

			px_logit = tf.contrib.layers.flatten(px_logit)


			output = tf.layers.dense(inputs=px_logit,
				                     units=self.x_flat_dim,
				                     activation=tf.nn.sigmoid,
				                     reuse=reuse,
				                     name='px_output')

		return px_logit, output


	def build_network(self):


		self.qy_logit, self.qy, self.y = self.Qy_graph(self.x_batch)

		self.z, self.zm, self.zv = self.Qz_graph(self.x_batch, self.y)
		self.zm_prior, self.zv_prior = self.Pz_graph(self.y)
		self.px_logit, self.output = self.Px_graph(self.z)

	def build_loss_optimizer(self):

		print('Defining loss function and optimizer...')
		with tf.name_scope('neg_entropy'):
			log_q = tf.nn.log_softmax(self.qy_logit)
			#self.nent = tf.nn.softmax_cross_entropy_with_logits(labels=self.qy, logits=self.qy_logit)
			self.nent = -tf.reduce_sum(self.qy*log_q, 1)
			#self.nent = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.qy, logits=self.qy_logit)

		nent = self.nent*self.r_nent

		self.loss_labeled = self.labeled_loss(self.K, self.z, self.zm, self.zv, self.zm_prior, self.zv_prior)

		#self.loss_recons = -self.log_bernoulli_with_logits(self.x_batch_flat, self.px_logit)
		loss_labeled = self.loss_labeled*self.r_label

		self.loss_recons = self.mean_squared_error(self.x_batch_flat, self.output)

		loss_recons = self.r_recons*self.loss_recons

		with tf.name_scope('final_loss'):
			self.loss = tf.add_n([-nent] + [loss_labeled] + [loss_recons])


		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

	def labeled_loss(self, k, z, zm, zv, zm_prior, zv_prior):
		""" Variational loss for the mixture of VAE
		"""
		return self.log_normal(z,zm,zv)-self.log_normal(z,zm_prior,zv_prior)-np.log(1/k)

	def log_normal(self, x, mu, var, eps=0.0, axis=-1):
		if eps > 0.0:
			var = tf.add(var, eps, name='clipped_var')
		return -0.5 * tf.reduce_sum(tf.log(2*np.pi)+tf.log(var)+tf.square(x-mu)/var, axis)

	def fit_batch(self, session, x, drop_rate=0.):
		tensors = [self.train_step, self.loss, self.nent, self.loss_labeled, self.loss_recons]
		feed_dict = {self.x_batch: x, self.drop_rate:drop_rate}
		_, loss, nent, labeled_loss, recons_loss = session.run(tensors, feed_dict=feed_dict)
		return loss, nent, labeled_loss, recons_loss

	def eval_batch(self, session, x, drop_rate=0.):
		tensors = [self.train_step, self.loss, self.nent, self.loss_labeled, self.loss_recons]
		feed_dict = {self.x_batch:x, self.drop_rate:drop_rate}
		_, loss, nent, labeled_loss, recons_loss = session.run(tensors, feed_dict=feed_dict)
		return loss, nent, labeled_loss, recons_loss


	def log_bernoulli_with_logits(self, x, logits, eps=0.0, axis=-1):
		#if eps > 0.0:
		#	max_val = np.log(1.0-eps)-np.log(eps)
		#	logits = tf.clip_by_value(logits, - max_val, max_val, name='clipped_logit')

		return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

	def mean_squared_error(self, real, predictions, average=False):

		loss = tf.square(real-predictions)
		if average:
			return tf.reduce_mean(loss)
		else:
			return tf.reduce_sum(loss,axis=-1)

	def sample_gumbel(self, shape):
		""" sample from Gumbel(0,1)

		Args:
			shape: (array) containing dimensions of the specified sample
		"""
		U = tf.random_uniform(shape, minval=0, maxval=1)
		return -tf.log(-tf.log(U+1e-6)+1e-6)

	def gumbel_softmax(self, logits, temperature, hard=False):
		""" sample from the gumbel-softmax distribution

		args:
			logits(array): [batch_size, n_class] unnormalized log-probs
			temperature(float): non-negative scalar
			hard: if True take argmax, but differentiate with respect to soft sample y

		returns:
			y: [batch_size, n_class] sample from the gumbel-softmax distribution
			If true the returned sample will be one-hot, otherwise it will be a probability
			distribution that sums to 1 across classes
		"""
		gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
		y = tf.nn.softmax(gumbel_softmax_sample/self.temperature)
		if hard:
			k = tf.shape(logits)[-1]
			y_hard = tf.cast(tf.equal(y, tf.reduce_sum(y, 1, keep_dims=True)), y.dtype)
			y = tf.stop_gradient(y_hard-y)+y
		return y


	def encode_y(self, session, x):
		feed = {self.x_batch:x, self.drop_rate:0.}
		tensors = [self.qy]
		return session.run(tensors, feed_dict=feed)

	def encode_zs(self, session, x):
		feed = {self.x_batch:x, self.drop_rate:0.}
		tensors = [self.z]
		return session.run(tensors, feed_dict=feed)

	def encode_zm(self, session, x):
		feed = {self.x_batch:x, self.drop_rate:0.}
		tensors = [self.zm]
		return session.run(tensors, feed_dict=feed)


	def reconstruct_xs(self, session, x):
		feed = {self.x_batch:x, self.drop_rate:0.}
		tensors = [self.xm]
		return session.run(tensors, feed_dict=feed)


	def generate_samples(self, session, category):
		feed = {self.x_batch: np.zeros(([self.batch_size, self.width, self.height, 1])), self.drop_rate:0.}


		indices = (np.ones(self.batch_size)*category).astype(int).tolist()
		categorical = tf.one_hot(indices, self.K)
		mean, var = self.Pz_graph(categorical)

		gaussian = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
		px, rec = self.Px_graph(gaussian)

		rec = tf.nn.sigmoid(px)
		return session.run(rec, feed_dict=feed)

