import os
import tensorflow as tf


class BaseModel:
	def __init__(self, checkpoint_dir, summary_dir, result_dir):
		
		self.checkpoint_dir = checkpoint_dir
		self.summary_dir = summary_dir
		self.result_dir = result_dir


	# save function that saves the checkpoint in the path defined in the config file
	def save(self, sess, saver, global_step_tensor):
		print('saving model...')
		saver.save(sess, self.checkpoint_dir + '/', global_step_tensor)
		print('Model saved')

	# load latest checkpoint from the experiment path defined in the config file
	def load(self, sess, saver):
		retval = False
		latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
		if latest_checkpoint:
			print('loading model checkponit {}...\n'.format(latest_checkpoint))
			saver.restore(sess, latest_checkpoint)
			print('model loaded')
			retval = True
		else:
			print('Model does not exist')
		return retval

