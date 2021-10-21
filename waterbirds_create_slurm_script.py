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
import copy
from pathlib import Path
import argparse
import numpy as np
import tqdm

import shared.utils as utils
import shared.cross_validation as cv

from waterbirds import configurator


BASE_DIR = '/data/ddmg/slabs/waterbirds'
FINAL_MODELS_DIR = f'{BASE_DIR}/final_models'

HOST = socket.gethostname()

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5] if HOST == 'milo' else [0, 1, 2, 3]
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
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	if not os.path.exists(hash_dir):
		print(hash_dir)
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
	config['gpuid'] = '$CUDA_VISIBLE_DEVICES'
	flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	if os.path.exists(f'/data/ddmg/slabs/structured_labels/slurm_scripts/{hash_string}.sbatch'):
		os.remove(f'/data/ddmg/slabs/structured_labels/slurm_scripts/{hash_string}.sbatch')
	f = open(f'/data/ddmg/slabs/structured_labels/slurm_scripts/{hash_string}.sbatch', 'x')
	f.write('#!/bin/bash\n')
	f.write('#SBATCH --time=8:00:00\n')
	f.write('#SBATCH --output=gpu.out\n')
	f.write('#SBATCH --partition=gpu\n')
	f.write('#SBATCH --gres=gpu:1 \n')
	f.write('#SBATCH --mem=9216MB \n')
	f.write('#SBATCH -w, --nodelist=vcuda-3,tig-slurm-2 \n')
	f.write(f'if [ ! -f "{hash_dir}/performance.pkl" ]; then\n')
	f.write(f'	python -m waterbirds.main {flags} > {hash_dir}/log.log 2>&1 \n')
	f.write('fi\n')
	f.close()

def main(experiment_name,
					model_to_tune,
					batch_size,
					oracle_prop,
					num_trials,
					overwrite,
					clean_back,
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
	all_config = configurator.get_sweep(experiment_name, model_to_tune, batch_size,
		clean_back, oracle_prop)
	print(f'All configs are {len(all_config)}')



	if not overwrite:
		configs_to_consider = [
			not utils.tried_config(config, base_dir=BASE_DIR) for config in all_config
		]
		all_config = list(itertools.compress(all_config, configs_to_consider))

	print(f'Remaining configs are {len(all_config)}')

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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--experiment_name', '-experiment_name',
		default='5050',
		choices=['5050', '5090', '8090', '8050', '8090_asym'],
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
			'oracle_aug', 'weighted_oracle_aug',
			'random_aug', 'weighted_random_aug',
			'rex'
			],
		help="Which model to tune",
		type=str)

	parser.add_argument('--oracle_prop', '-oracle_prop',
		default=-1.0,
		help=("Proportion of training data to use for augentation."
					"Only relevant if model_to_tune is [something]_aug"),
		type=float)

	parser.add_argument('--num_trials', '-num_trials',
		default=1e6,
		help="Number of hyperparameters to try",
		type=int)

	parser.add_argument('--batch_size', '-batch_size',
		default=64,
		help=("training batch size"),
		type=int)

	parser.add_argument('--overwrite', '-overwrite',
		action='store_true',
		default=False,
		help="If this config has been tested before, rerun?")

	parser.add_argument('--clean_back', '-clean_back',
		default='True',
		help="Run with clean or noisy backgrounds",
		type=str)

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
