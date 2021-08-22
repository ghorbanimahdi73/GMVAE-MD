import tensorflow as tf
import os

class Logger:
	def __init__(self, sess, summary_dir):
		self.sess = sess
		self.summary_dir = summary_dir
		self.summary_placeholders = {}
		self.summary_ops = {}
		self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, 'train'), self.sess.graph)
		self.test_summary_writer  = tf.summary.FileWriter(os.path.join(self.summary_dir, 'test'))

	def summarize(self, step, summarizer='train', scope='', summaries_dict=None):
		"""
		:param step: the step of the summary
		:param summarizer: use the train summary writer or the test one
		:param scope: variable scope
		:param summaries_dict: the dict of the summaries values
		:return
		"""
		summary_writer = self.train_summary_writer if summarizer=='train' else self.test_summary_writer
		with tf.variable_scope(scope):
			if summaries_dict is not None:
				summary_list = []
				for tag, value in summaries_dict.items():
					if len(value.shape) <= 1:
						self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
					else:
						self.summary_placeholders[tag] = tf.placeholder('float32', [None]+list(value.shape[1:]), name=tag)
					if len(value.shape) <= 1:
						self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
					else:
						self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

				summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]:value}))

			for summary in summary_list:
				summary_writer.add_summary(summary, step)
			summary_writer.flush()
