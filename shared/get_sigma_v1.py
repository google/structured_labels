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
""" Cross validation algorithms for slabs and benchmarks. """

import functools
import logging
import multiprocessing
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.stats import ttest_1samp
import tqdm

from shared.utils import config_hasher, tried_config_file
from waterbirds import configurator

NUM_WORKERS = 10
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

def import_helper(config, base_dir):
	"""Imports the dictionary with the results of an experiment.

	Args:
		args: tuple with model, config where
			model: str, name of the model we're importing the performance of
			config: dictionary, expected to have the following: exp_dir, the experiment
				directory random_seed,  random seed for the experiment py1_y0_s,
				probability of y1=1| y0=1 in the shifted test distribution alpha,
				MMD/cross prediction penalty sigma,  kernel bandwidth for the MMD penalty
				l2_penalty,  regularization parameter dropout_rate,  drop out rate
				embedding_dim,  dimension of the final representation/embedding
				unused_kwargs, other key word args passed to xmanager but not needed here

	Returns:
		pandas dataframe of results if the file was found, none otherwise
	"""
	if config is None:
		return
	hash_string = config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')

	if not os.path.exists(performance_file):
		logging.error('Couldnt find %s', performance_file)
		return None

	results_dict = pickle.load(open(performance_file, 'rb'))
	results_dict.update(config)
	return pd.DataFrame(results_dict, index=[0])


def import_results(configs, base_dir):
	tried_config_wrapper = functools.partial(tried_config_file, base_dir=base_dir)

	available_configs = []
	pool = multiprocessing.Pool(NUM_WORKERS)
	for ac in tqdm.tqdm(pool.map(tried_config_wrapper, configs),
		total=len(configs)):
		available_configs.append(ac)

	import_helper_wrapper = functools.partial(import_helper, base_dir=base_dir)
	pool = multiprocessing.Pool(NUM_WORKERS)
	res = []
	for config_res in tqdm.tqdm(pool.imap_unordered(import_helper_wrapper,
		available_configs), total=len(available_configs)):
		res.append(config_res)
	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	return res


def get_sigma(configs, base_dir, hparams):
	all_results = import_results(configs, base_dir)
	all_results = all_results[(all_results.l2_penalty == 0.0)]
	columns_to_keep = [col for col in all_results.columns if 'validation_mmd' in col]
	columns_to_keep = ['random_seed'] + columns_to_keep

	all_results = all_results[columns_to_keep]
	all_results.drop('validation_mmd', axis=1, inplace=True)
	all_results.rename(columns={'validation_mmd1': 'validation_mmd1.0'},
		inplace=True)

	all_results_long = []
	for col in all_results.columns:
		if col == 'random_seed':
			continue
		sigma_df = all_results[['random_seed', col]]

		sigma_value = float(re.findall("\d+\.\d+", col)[0])
		sigma_df.columns = ['random_seed', 'validation_mmd']
		sigma_df['sigma'] = sigma_value
		all_results_long.append(sigma_df)

	all_results_long = pd.concat(all_results_long, axis=0, ignore_index=True)
	print(all_results_long[(all_results_long.random_seed==0)])
	all_results_long = all_results_long[(all_results_long.validation_mmd <= 1e-2)]
	all_results_long = all_results_long.groupby('random_seed').sigma.min()
	print(all_results_long)

if __name__ == "__main__":
	experiment_name = "8090"
	model_to_tune = "weighted_baseline"
	oracle_prop = 0.0
	all_config = configurator.get_sweep(experiment_name, model_to_tune, oracle_prop)
	get_sigma(all_config, BASE_DIR, ['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
				'embedding_dim'])



