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

"""Main script for chexpert experiment."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
import functools
import itertools
import subprocess
import socket
import multiprocessing
import pickle
import copy
from pathlib import Path
import argparse
import numpy as np
import tqdm

import shared.utils as utils
import shared.cross_validation as cv
import shared.post_training_eval as post_training_eval

from chexpert import configurator


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'chexpert'))
FINAL_MODELS_DIR = f'{BASE_DIR}/final_models'

HOST = socket.gethostname()

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7] if HOST == 'milo' else [0, 1, 2, 3]
if HOST not in  ['milo', 'ahoy', 'mars', 'twix', 'oreo']:
	AVAILABLE_GPUS = os.environ['CUDA_VISIBLE_DEVICES']
	AVAILABLE_GPUS = [gpu for gpu in AVAILABLE_GPUS if gpu != ',' and gpu !=' ']

NUM_GPUS = len(AVAILABLE_GPUS)
PROC_PER_GPU = 1
NUM_DELETE_WORKERS = 20
PVAL= 0.05

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

	# output = os.statvfs('/data/scratch/mmakar')
	# avail = output.f_frsize * output.f_bavail
	# total = output.f_frsize * output.f_blocks
	# prop_avail = avail / total
	# if prop_avail < 0.1:
	# 	raise ValueError("Running low on scratch space")


	try:
		hash_string = utils.config_hasher(config)
		hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
		if (not overwrite) and utils.tried_config(config, BASE_DIR):
			return None
		if not os.path.exists(hash_dir):
			os.system(f'mkdir -p {hash_dir}')

		if (('warmstart_dir' in config.keys()) and (config['warmstart_dir'] == 'find')):
			try:
				warmstart_config = copy.deepcopy(config)
				warmstart_config['weighted_mmd'] = 'False'
				warmstart_config['balanced_weights'] = 'False'
				del warmstart_config['warmstart_dir']
				warmstart_hash_string =  utils.config_hasher(warmstart_config)
				warmstart_hash_dir = os.path.join(BASE_DIR, 'tuning', warmstart_hash_string, 'saved_model')
				subdirs = [x for x in Path(warmstart_hash_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
				warmstart_hash_dir = str(sorted(subdirs)[-1])
				config['warmstart_dir'] = f'{warmstart_hash_dir}/variables/variables'
			except:
				raise NotImplementedError("not yet")

		config['exp_dir'] = hash_dir
		config['cleanup'] = True

		chosen_gpu = QUEUE.get()
		config['gpuid'] = chosen_gpu
		flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
		subprocess.call(f'python -m chexpert.main {flags} > {hash_dir}/log.log 2>&1',
			shell=True)
		# subprocess.call('python -m chexpert.main %s' % flags, shell=True)
		# print(f'python -m chexpert.main {flags} > {hash_dir}/log.log 2>&1')
		config.pop('exp_dir')
		config.pop('cleanup')
		pickle.dump(config, open(os.path.join(hash_dir, 'config.pkl'), 'wb'))
	finally:
		QUEUE.put(chosen_gpu)


def main(experiment_name,
					model_to_tune,
					num_trials,
					overwrite,
					train_models,
					pick_best,
					clean_directories):
	"""Main function to tune/train the model.
	Args:
		experiment_name: str, name of the experiemnt to run
		model_to_tune: str, which model to tune/train
		oracle_prop: float, proportion to use for training augmentation. Only relevant
				if model_to_tune is [something]_aug
		num_trials: int, number of hyperparams to train for
		num_workers: int, number of workers to run in parallel
		overwrite: bool, whether or not to retrain if a specific hyperparam config
			has already been tried

		Returns:
			nothing
	"""
	all_config = configurator.get_sweep(experiment_name, model_to_tune)
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

		original_configs = len(all_config)
		configs_available = [
			utils.tried_config(config, base_dir=BASE_DIR) for config in all_config
		]
		all_config = list(itertools.compress(all_config, configs_available))
		found_configs = len(all_config)
		print(f'------ FOUND {found_configs} / {original_configs}---------')


		if not os.path.exists(FINAL_MODELS_DIR):
			os.mkdir(FINAL_MODELS_DIR)

		if 'rex' in model_to_tune:
			classic_final_model, classic_final_optimal_config = \
				cv.get_optimal_model_results(mode='accuracy', configs=all_config,
					base_dir=BASE_DIR, hparams=['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
					'embedding_dim'], weighted_xv='False')

			classic_final_model['model'] = f'{model_to_tune}_classic'
			classic_final_model.to_csv(
				f'{FINAL_MODELS_DIR}/{model_to_tune}_classic_{experiment_name}.csv',
				index=False)
			classic_final_optimal_config.to_csv(
				(f'{FINAL_MODELS_DIR}/optimal_config_{model_to_tune}_classic_'
					f'{experiment_name}.csv'),
				index=False)

		else:

			classic_final_model, classic_final_optimal_config = \
				cv.get_optimal_model_results(mode='classic', configs=all_config,
					base_dir=BASE_DIR, hparams=['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
					'embedding_dim'], weighted_xv='False')

			classic_final_model['model'] = f'{model_to_tune}_classic'
			classic_final_model.to_csv(
				f'{FINAL_MODELS_DIR}/{model_to_tune}_classic_{experiment_name}.csv',
				index=False)

			classic_final_optimal_config.to_csv(
				(f'{FINAL_MODELS_DIR}/optimal_config_{model_to_tune}_classic_'
					f'{experiment_name}.csv'),
				index=False)

			if (model_to_tune == 'slabs_weighted_bal') or (model_to_tune == 'slabs_unweighted_two_way') :
				print("===== 2 step xv======")
				twostep_final_model, twostep_final_optimal_config = \
					cv.get_optimal_model_results(mode='two_step', configs=all_config,
						base_dir=BASE_DIR, hparams=['alpha', 'sigma',
						'dropout_rate', 'l2_penalty', 'embedding_dim'],
						weighted_xv='weighted_bal', pval=PVAL)

				twostep_final_model['model'] = f'{model_to_tune}_ts{PVAL}'
				twostep_final_model.to_csv(
					f'{FINAL_MODELS_DIR}/{model_to_tune}_ts{PVAL}_{experiment_name}.csv',
					index=False)

				twostep_final_optimal_config.to_csv(
					(f'{FINAL_MODELS_DIR}/optimal_config_{model_to_tune}_ts{PVAL}'
						f'_{experiment_name}.csv'),
					index=False)

			if model_to_tune == 'unweighted_slabs':
				twostep_final_model, twostep_final_optimal_config = \
					cv.get_optimal_model_results(mode='two_step', configs=all_config,
						base_dir=BASE_DIR, hparams=['alpha', 'sigma', 'dropout_rate',
						'l2_penalty', 'embedding_dim'], weighted_xv='False', pval=PVAL)

				twostep_final_model['model'] = f'{model_to_tune}_uts{PVAL}'
				twostep_final_model.to_csv(
					f'{FINAL_MODELS_DIR}/{model_to_tune}_uts{PVAL}_{experiment_name}.csv',
					index=False)

				twostep_final_optimal_config.to_csv(
					(f'{FINAL_MODELS_DIR}/optimal_config_{model_to_tune}_uts{PVAL}'
						f'_{experiment_name}.csv'),
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
		default='unskew_train',
		choices=['unskew_train', 'skew_train'],
		help="Which experiment to run",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='slabs',
		choices=[
			'slabs_weighted', 'slabs_weighted_bal', 'slabs_weighted_bal_two_way',
			'slabs_warmstart_weighted', 'slabs_warmstart_weighted_bal',
			'slabs_logit', 'slabs_unweighted_two_way',
			'unweighted_slabs', 'unweighted_slabs_logit',
			'simple_baseline','weighted_baseline',
			'random_aug', 'weighted_random_aug', 'rex'],
		help="Which model to tune",
		type=str)

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
