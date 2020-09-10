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

"""Runs the full correlation sweep for the corrupted mnist experiment."""

import hashlib
from itertools import compress
from subprocess import call
from multiprocessing import Pool
import numpy as np
import os


from argparse import ArgumentParser
import pickle
import tqdm

from cmnist import configurator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))


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


def runner(config):
	"""Trains model in config if not trained before.
	Args:
		config: dict with config
	Returns:
		Nothing
	"""
	hash_string = config_hasher(config)
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	if not os.path.exists(hash_dir):
		os.system(f'mkdir -p {hash_dir}')
	config['exp_dir'] = hash_dir
	flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	call('python -m cmnist.main %s > /dev/null 2>&1' % flags, shell=True)
	# call('python -m cmnist.main %s' % flags, shell=True)
	config.pop('exp_dir')
	pickle.dump(config, open(os.path.join(hash_dir, 'config.pkl'), 'wb'))


def main(experiment_name, model_to_tune, num_trials, num_workers, overwrite):
	"""Main function to tune/train the model.
	Args:
		experiment_name: str, name of the experiemnt to run
		model_to_tune: str, which model to tune/train
		num_trials: int, number of hyperparams to train for
		num_workers: int, number of workers to run in parallel
		overwrite: bool, whether or not to retrain if a specific hyperparam config
			has already been tried

		Returns:
			nothing
	"""
	all_config = configurator.get_sweep(experiment_name, model_to_tune)
	if not overwrite:
		configs_to_consider = [not tried_config(config, base_dir=BASE_DIR) for config
												in all_config]
		all_config = list(compress(all_config, configs_to_consider))

	if num_trials < len(all_config):
		configs_to_run = np.random.choice(len(all_config), size=num_trials,
			replace=False).tolist()
		configs_to_run = [config_id in configs_to_run for config_id in
			range(len(all_config))]
		all_config = list(compress(all_config, configs_to_run))

	assert len(all_config) <= num_trials
	if num_workers > 1:
		pool = Pool(num_workers)
		for _ in tqdm.tqdm(pool.imap_unordered(runner, all_config),
			total=len(all_config)):
			pass
	else:
		for config in all_config:
			runner(config)


if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument('--experiment_name', '-experiment_name',
		default='correlation',
		choices=['correlation', 'overlap'],
		help="Which experiment to run",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='slabs',
		choices=['slabs', 'opslabs', 'simple_baseline'],
		help="Which model to tune",
		type=str)

	parser.add_argument('--num_trials', '-num_trials',
		default=5000,
		help="Number of hyperparameters to try",
		type=int)

	parser.add_argument('--num_workers', '-num_workers',
		default=20,
		help="Number of workers to run in parallel",
		type=int)

	parser.add_argument('--overwrite', '-overwrite',
		default=False,
		help="If this config has been tested before, rerun?",
		type=bool)

	args = vars(parser.parse_args())
	main(**args)
