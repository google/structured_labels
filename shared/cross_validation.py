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

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.stats import ttest_1samp
import tqdm

from shared.utils import config_hasher, tried_config_file

NUM_WORKERS = 10


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


def reshape_results(results):
	shift_columns = [col for col in results.columns if 'shift' in col]
	shift_metrics_columns = [
		col for col in shift_columns if ('pred_loss' in col) or ('accuracy' in col) or ('auc' in col)
	]
	results = results[shift_metrics_columns]
	results = results.transpose()
	results['py1_y0_s'] = results.index.str[6:10]
	results['py1_y0_s'] = results.py1_y0_s.str.replace('_', '')
	results['py1_y0_s'] = results.py1_y0_s.astype(float)

	results_accuracy = results[(results.index.str.contains('accuracy'))]
	results_accuracy = results_accuracy.rename(columns={
		col: f'accuracy_{col}' for col in results_accuracy.columns if col != 'py1_y0_s'
	})

	results_auc = results[(results.index.str.contains('auc'))]
	results_auc = results_auc.rename(columns={
		col: f'auc_{col}' for col in results_auc.columns if col != 'py1_y0_s'
	})
	results_loss = results[(results.index.str.contains('pred_loss'))]
	results_loss = results_loss.rename(columns={
		col: f'loss_{col}' for col in results_loss.columns if col != 'py1_y0_s'
	})

	results_final = results_accuracy.merge(results_loss, on=['py1_y0_s'])
	results_final = results_final.merge(results_auc, on=['py1_y0_s'])
	print(results_final)
	return results_final


def get_optimal_model_results(mode, configs, base_dir, hparams,
	equivalent=True, pval=False):

	if mode not in ['classic', 'two_step']:
		raise NotImplementedError('Can only run classic or two_step modes')
	if mode == 'classic':
		return get_optimal_model_classic(configs, None, base_dir, hparams)
	return get_optimal_model_two_step(configs, base_dir, hparams)


def get_optimal_model_two_step(configs, base_dir, hparams, epsilon=1e-2):
	all_results = import_results(configs, base_dir)
	# -- get those with validation mmd < epsilon
	columns_to_keep = hparams + ['random_seed', 'validation_mmd']
	best_mmd = all_results[columns_to_keep]
	# for runs where the min validation mmd > epsilon, use their min possible
	best_mmd = best_mmd.groupby('random_seed').validation_mmd.min()
	best_mmd = best_mmd.to_frame()

	best_mmd.rename(columns={'validation_mmd': 'min_validation_mmd'},
		inplace=True)
	filtered_results = all_results.merge(best_mmd, on='random_seed')

	# filtered_results = filtered_results[
	# 	(filtered_results.validation_mmd - filtered_results.min_validation_mmd) <= epsilon
	# ]

	filtered_results = filtered_results[(filtered_results.validation_mmd <= epsilon)]
	# filtered_results['min_validation_mmd'] = np.where(
	# 	filtered_results['min_validation_mmd'] > epsilon, 
	# 	epsilon, filtered_results['min_validation_mmd']
	# )

	filtered_results.reset_index(drop=True, inplace=True)

	filtered_results.drop('min_validation_mmd', axis = 1, inplace=True)
	assert len(list(set(filtered_results.columns.tolist()) - set(all_results.columns.tolist()))) == 0
	assert len(list(set(all_results.columns.tolist()) - set(filtered_results.columns.tolist()))) == 0


	return get_optimal_model_classic(None, filtered_results, base_dir, hparams)


def get_optimal_model_classic(configs, filtered_results, base_dir, hparams):
	if ((configs is None) and (filtered_results is None)):
		raise ValueError("Need either configs or table of results_dict")

	if configs is not None:
		all_results = import_results(configs, base_dir)
	else:
		all_results = filtered_results.copy()


	# ---get optimal hyperparams based on prediction loss
	columns_to_keep = hparams + ['random_seed', 'validation_pred_loss']
	best_loss = all_results[columns_to_keep]
	best_loss = best_loss.groupby('random_seed').validation_pred_loss.min()
	best_loss = best_loss.to_frame()

	best_loss.reset_index(drop=False, inplace=True)
	best_loss.rename(columns={'validation_pred_loss': 'min_validation_pred_loss'},
		inplace=True)
	all_results = all_results.merge(best_loss, on='random_seed')
	all_results = all_results[
		(all_results.validation_pred_loss == all_results.min_validation_pred_loss)
	]

	# --- get the final results over all runs
	mean_results = all_results.mean(axis=0).to_frame()
	mean_results.rename(columns={0: 'mean'}, inplace=True)
	std_results = all_results.std(axis=0).to_frame()
	std_results.rename(columns={0: 'std'}, inplace=True)
	final_results = mean_results.merge(
		std_results, left_index=True, right_index=True
	)

	final_results = final_results.transpose()
	final_results_clean = reshape_results(final_results)

	return final_results_clean, None

