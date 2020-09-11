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
