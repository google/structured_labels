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
import functools
import itertools
import subprocess
import multiprocessing
import os
import pickle

import argparse
import numpy as np  # noqa: F401
import tqdm

from shared.utils import config_hasher, tried_config
from cmnist import configurator


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))
NUM_GPUS = 4
PROC_PER_GPU = 3

QUEUE = multiprocessing.Queue()
for gpu_ids in range(NUM_GPUS):
	for _ in range(PROC_PER_GPU):
		QUEUE.put(str(gpu_ids))


def runner(config, overwrite):
	"""Trains model in config if not trained before.
	Args:
		config: dict with config
	Returns:
		Nothing
	"""
	try:
		hash_string = config_hasher(config)
		hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
		if (not overwrite) and tried_config(config, BASE_DIR):
			return None
		if not os.path.exists(hash_dir):
			os.system(f'mkdir -p {hash_dir}')
		config['exp_dir'] = hash_dir
		config['cleanup'] = True
		# chosen_gpu = get_gpu_assignment()
		chosen_gpu = QUEUE.get()
		config['gpuid'] = chosen_gpu
		flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
		subprocess.call('python -m cmnist.main %s > /dev/null 2>&1' % flags,
			shell=True)
		# subprocess.call('python -m cmnist.main %s' % flags, shell=True)
		config.pop('exp_dir')
		config.pop('cleanup')
		pickle.dump(config, open(os.path.join(hash_dir, 'config.pkl'), 'wb'))
	finally:
		QUEUE.put(chosen_gpu)


def main(experiment_name,
					model_to_tune,
					aug_prop,
					num_trials,
					overwrite):
	"""Main function to tune/train the model.
	Args:
		experiment_name: str, name of the experiemnt to run
		model_to_tune: str, which model to tune/train
		aug_prop: float, proportion to use for training augmentation. Only relevant
				if model_to_tune is [something]_aug
		num_trials: int, number of hyperparams to train for
		num_workers: int, number of workers to run in parallel
		overwrite: bool, whether or not to retrain if a specific hyperparam config
			has already been tried

		Returns:
			nothing
	"""
	all_config = configurator.get_sweep(experiment_name, model_to_tune, aug_prop)
	print(f'All configs are {len(all_config)}')
	if not overwrite:
		configs_to_consider = [not tried_config(config, base_dir=BASE_DIR) for config
												in all_config]
		all_config = list(itertools.compress(all_config, configs_to_consider))

	if num_trials < len(all_config):
		configs_to_run = np.random.choice(len(all_config), size=num_trials,
			replace=False).tolist()
		configs_to_run = [config_id in configs_to_run for config_id in
			range(len(all_config))]
		all_config = list(itertools.compress(all_config, configs_to_run))

	pool = multiprocessing.Pool(NUM_GPUS * PROC_PER_GPU)
	runner_wrapper = functools.partial(runner, overwrite=overwrite)
	for _ in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
		total=len(all_config)):
		pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--experiment_name', '-experiment_name',
		default='correlation',
		choices=['correlation', 'overlap'],
		help="Which experiment to run",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='slabs',
		choices=['slabs', 'opslabs', 'weighted_opslabs', 'simple_baseline',
		'oracle_aug'],
		help="Which model to tune",
		type=str)

	parser.add_argument('--aug_prop', '-aug_prop',
		default=-1.1,
		help=("Proportion of training data to use for augentation."
					"Only relevant if model_to_tune is [something]_aug"),
		type=float)

	parser.add_argument('--num_trials', '-num_trials',
		default=1e6,
		help="Number of hyperparameters to try",
		type=int)

	parser.add_argument('--overwrite', '-overwrite',
		default=False,
		help="If this config has been tested before, rerun?",
		type=bool)

	args = vars(parser.parse_args())
	main(**args)
