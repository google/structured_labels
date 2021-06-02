# Copyright 2020 the Causally Motivated Shortcut Removal
# Authors. All rights reserved.

"""Utility functions to support the main training algorithm"""
import glob
import os
import tensorflow as tf

def extract_weights(labels, params):
	""" Extracts the weights from the labels dictionary. """

	sample_weights_pos = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 0]), axis=-1)
	sample_weights_neg = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 1]), axis=-1)
	sample_weights = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 2]), axis=-1)
	return sample_weights, sample_weights_pos, sample_weights_neg

def flatten_dict(dd, separator='_', prefix=''):
	""" Flattens the dictionary with eval metrics """
	return {
		prefix + separator + k if prefix else k: v
		for kk, vv in dd.items()
		for k, v in flatten_dict(vv, separator, kk).items()
	} if isinstance(dd,
		dict) else {prefix: dd}


def cleanup_directory(directory):
	"""Deletes all files within directory and its subdirectories.

	Args:
		directory: string, the directory to clean up
	Returns:
		None
	"""
	if os.path.exists(directory):
		files = glob.glob(f"{directory}/*", recursive=True)
		for f in files:
			if os.path.isdir(f):
				os.system(f"rm {f}/* ")
			else:
				os.system(f"rm {f}")


