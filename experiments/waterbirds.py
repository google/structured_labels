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
import socket
import multiprocessing
import os
import pickle

import argparse
import numpy as np
import tqdm

import shared.utils as utils
import shared.cross_validation as cv

from waterbirds import configurator


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))
FINAL_MODELS_DIR = f'{BASE_DIR}/final_models'

HOST = socket.gethostname()

AVAILABLE_GPUS = [i for i in range(5)] if HOST == 'milo' else [0, 1, 2]
NUM_GPUS = len(AVAILABLE_GPUS)
PROC_PER_GPU = 1 if HOST == 'milo' else 1
NUM_DELETE_WORKERS = 20

QUEUE = multiprocessing.Queue()
for gpu_ids in AVAILABLE_GPUS:
	for _ in range(PROC_PER_GPU):
		QUEUE.put(str(gpu_ids))


def runner(config, overwrite):
	"""Trains model in config if not trained before.
	Args:
		config: dict with config
	Returns:
		Nothing
	"""
	# check system status
	output = os.statvfs('/data/ddmg/slabs/')
	avail = output.f_frsize * output.f_bavail
	total = output.f_frsize * output.f_blocks
	prop_avail = avail / total
	if prop_avail < 0.1:
		raise ValueError("Running low on NFS space")
	try:
		hash_string = utils.config_hasher(config)
		hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
		if (not overwrite) and utils.tried_config(config, BASE_DIR):
			return None
		if not os.path.exists(hash_dir):
			os.system(f'mkdir -p {hash_dir}')
		config['exp_dir'] = hash_dir
		config['cleanup'] = True
		chosen_gpu = QUEUE.get()
		config['gpuid'] = chosen_gpu
		flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
		subprocess.call(f'python -m waterbirds.main {flags} > {hash_dir}/log.log 2>&1',
			shell=True)
		# subprocess.call('python -m waterbirds.main %s' % flags, shell=True)
		# print(f'python -m waterbirds.main %s > /dev/null 2>&1' % flags)
		config.pop('exp_dir')
		config.pop('cleanup')
		pickle.dump(config, open(os.path.join(hash_dir, 'config.pkl'), 'wb'))
	finally:
		QUEUE.put(chosen_gpu)


def main(experiment_name,
					model_to_tune,
					aug_prop,
					num_trials,
					overwrite,
					train_models,
					pick_best,
					clean_directories):
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


	if train_models:
		if not overwrite:
			configs_to_consider = [
				not utils.tried_config(config, base_dir=BASE_DIR) for config in all_config
			]
			all_config = list(itertools.compress(all_config, configs_to_consider))

		if num_trials < len(all_config):
			configs_to_run = np.random.choice(len(all_config), size=num_trials,
				replace=False).tolist()
			configs_to_run = [config_id in configs_to_run for config_id in
				range(len(all_config))]
			all_config = list(itertools.compress(all_config, configs_to_run))
		# all_config = all_config[:1]
		pool = multiprocessing.Pool(NUM_GPUS * PROC_PER_GPU)
		runner_wrapper = functools.partial(runner, overwrite=overwrite)
		for _ in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
			total=len(all_config)):
			pass

	elif pick_best:
		if not os.path.exists(FINAL_MODELS_DIR):
			os.mkdir(FINAL_MODELS_DIR)

		classic_final_model, _ = \
			cv.get_optimal_model_results(mode='classic', configs=all_config,
				base_dir=BASE_DIR, hparams=['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
				'embedding_dim'], pval=False)

		classic_final_model['model'] = f'{model_to_tune}_classic'
		classic_final_model.to_csv(
			f'{FINAL_MODELS_DIR}/{model_to_tune}_classic_{experiment_name}.csv',
			index=False)

		if 'slabs' in model_to_tune:
			twostep_final_model, _ = \
				cv.get_optimal_model_results(mode='two_step', configs=all_config,
					base_dir=BASE_DIR, hparams=['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
					'embedding_dim'], pval=False)

			twostep_final_model['model'] = f'{model_to_tune}_ts'
			twostep_final_model.to_csv(
				f'{FINAL_MODELS_DIR}/{model_to_tune}_ts_{experiment_name}.csv',
				index=False)


	elif clean_directories:
		print("Are you sure you want to delete? Uncomment the next line then!")
		assert 1 == 2
		for _ in all_config:
			delete_config_file_wrapper = functools.partial(
				utils.delete_config_file, base_dir=BASE_DIR)

		pool = multiprocessing.Pool(NUM_DELETE_WORKERS)
		for _ in tqdm.tqdm(pool.imap_unordered(delete_config_file_wrapper,
			all_config), total=len(all_config)):
			pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--experiment_name', '-experiment_name',
		default='5050',
		choices=['5050', '5090', '8090'],
		help="Which experiment to run",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='slabs',
		choices=[
			'slabs', 'slabs_logit',
			'unweighted_slabs', 'unweighted_slabs_logit',
			'simple_baseline', 'weighted_baseline',
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
		action='store_true',
		default=False,
		help="If this config has been tested before, rerun?")

	parser.add_argument('--train_models', '-train_models',
		action='store_true',
		default=False,
		help="Train models for all configs?")

	parser.add_argument('--pick_best', '-pick_best',
		action='store_true',
		default=False,
		help="Pick the optimal model based on existing results?")

	parser.add_argument('--clean_directories', '-clean_directories',
		action='store_true',
		default=False,
		help="NUCLEAR: delete all model results?")

	args = vars(parser.parse_args())
	main(**args)
