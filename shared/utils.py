# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility functions."""

import hashlib
import os

import tensorflow as tf


def restrict_GPU_tf(gpuid, memfrac=0, use_cpu=False):
	""" Function to pick the gpu to run on
		Args:
			gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
			memfrac: float, fraction of memory. By default grows dynamically
	"""
	if not use_cpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

		config = tf.compat.v1.ConfigProto()
		if memfrac == 0:
			config.gpu_options.allow_growth = True
		else:
			config.gpu_options.per_process_gpu_memory_fraction = memfrac
		tf.compat.v1.Session(config=config)
		print("Using GPU:{} with {:.0f}% of the memory".format(gpuid, memfrac * 100))
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

		print("Using CPU")


def config_hasher(config):
	"""Generates hash string for a given config.
	Args:
		config: dict with hyperparams ordered by key
	Returns:
		hash of config
	"""
	config_string = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	hash_string = hashlib.sha256(config_string.encode()).hexdigest()
	return hash_string


def tried_config(config, base_dir):
	"""Tests if config has been tried before.
	Args:
		config: hyperparam config
		base_dir: directory where the tuning folder lives
	"""
	hash_string = config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')
	return os.path.isfile(performance_file)
