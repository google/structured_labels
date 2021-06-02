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
# import collections

from shared.utils import config_hasher, tried_config_file
from shared import get_sigma

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

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
	# print(results_dict['validation_global_step'])
	results_dict.update(config)
	if 'pflip0' not in results_dict.keys():
		results_dict['pflip0'] = 0.0
		results_dict['pflip1'] = 0.0
		results_dict['clean_back'] = 'NA'
		results_dict['py0'] = 'NA'
		results_dict['py1_y0'] = 'NA'
	return pd.DataFrame(results_dict, index=[0])


def import_results(configs, base_dir):
	# tried_config_wrapper = functools.partial(tried_config_file, base_dir=base_dir)

	# available_configs = []
	# pool = multiprocessing.Pool(NUM_WORKERS)
	# for ac in tqdm.tqdm(pool.map(tried_config_wrapper, configs),
	# 	total=len(configs)):
	# 	if ac != None:
	# 		available_configs.append(ac)

	# base_dir = '/data/ddmg/slabs/.zfs/snapshot/weekly-2021-16/waterbirds/'
	import_helper_wrapper = functools.partial(import_helper, base_dir=base_dir)
	pool = multiprocessing.Pool(NUM_WORKERS)
	res = []
	for config_res in tqdm.tqdm(pool.imap_unordered(import_helper_wrapper,
		configs), total=len(configs)):
		res.append(config_res)
	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	return res, configs


def reshape_results(results):
	shift_columns = [col for col in results.columns if 'shift' in col]
	shift_metrics_columns = [
		col for col in shift_columns if ('pred_loss' in col) or ('accuracy' in col) or ('auc' in col)
	]
	# shift_metrics_columns = shift_metrics_columns + [
	# 	f'shift_{py}_mmd' for py in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]]

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


	# results_mmd = results[(results.index.str.contains('mmd'))]
	# results_mmd = results_mmd.rename(columns={
	# 	col: f'mmd_{col}' for col in results_loss.columns if col != 'py1_y0_s'
	# })


	results_final = results_accuracy.merge(results_loss, on=['py1_y0_s'])
	# results_final = results_final.merge(results_mmd, on=['py1_y0_s'])
	results_final = results_final.merge(results_auc, on=['py1_y0_s'])


	print(results_final)
	return results_final


def get_optimal_model_results(mode, configs, base_dir, hparams,
	equivalent=True, weighted_xv=False, pval=0.1):

	if mode not in ['classic', 'two_step', 'three_step']:
		raise NotImplementedError('Can only run classic or two_step modes')
	if mode == 'classic':
		return get_optimal_model_classic(configs, None, base_dir, hparams)
	elif mode =='two_step':
		return get_optimal_model_two_step(configs, base_dir, hparams, weighted_xv, pval)
	elif mode == 'three_step':
		return get_optimal_model_three_step(configs, base_dir, hparams, weighted_xv, pval)



def get_optimal_model_two_step(configs, base_dir, hparams, weighted_xv, pval):
	all_results, available_configs = import_results(configs, base_dir)
	sigma_results = get_sigma.get_optimal_sigma(available_configs, kfolds=3,
		weighted_xv=weighted_xv, compute_loss=False)
	# print("this is sig res")
	print(sigma_results.sort_values(['random_seed', 'sigma', 'alpha']))
	best_pval = sigma_results.groupby('random_seed').pval.max()
	best_pval = best_pval.to_frame()
	best_pval.reset_index(inplace=True, drop=False)
	best_pval.rename(columns={'pval': 'best_pval'}, inplace=True)

	smallest_mmd = sigma_results.groupby('random_seed').mmd.min()
	smallest_mmd = smallest_mmd.to_frame()
	smallest_mmd.reset_index(inplace=True, drop=False)
	smallest_mmd.rename(columns={'mmd': 'smallest_mmd'}, inplace=True)

	sigma_results = sigma_results.merge(best_pval, on ='random_seed')
	sigma_results = sigma_results.merge(smallest_mmd, on ='random_seed')

	filtered_results = all_results.merge(sigma_results, on=['random_seed', 'sigma', 'alpha'])

	filtered_results = filtered_results[
		(((filtered_results.pval >= pval) &  (filtered_results.best_pval >= pval)) | \
		((filtered_results.best_pval < pval) &  (filtered_results.mmd == filtered_results.smallest_mmd)))
		]


	best_pval_by_seed = filtered_results[['random_seed', 'pval']].copy()
	best_pval_by_seed = best_pval_by_seed.groupby('random_seed').pval.min()
	# print("===Best pvalue=====")
	# print(best_pval_by_seed)



	filtered_results.drop(['pval', 'best_pval'], inplace=True, axis=1)
	filtered_results.reset_index(drop=True, inplace=True)

	unique_filtered_results = filtered_results[['random_seed', 'sigma', 'alpha']].copy()
	unique_filtered_results.drop_duplicates(inplace=True)

	# print("===valid hparams=====")
	# print(unique_filtered_results.sort_values(['random_seed', 'sigma', 'alpha']))
	# print(len(unique_filtered_results))


	return get_optimal_model_classic(None, filtered_results, base_dir, hparams)

def get_optimal_model_three_step(configs, base_dir, hparams, weighted_xv, pval):
	all_results, available_configs = import_results(configs, base_dir)
	sigma_results = get_sigma.get_optimal_sigma(available_configs,
		kfolds=5, weighted_xv=weighted_xv, compute_loss=True)

	filtered_results = all_results.merge(sigma_results, on=['random_seed', 'sigma', 'alpha'])
	filtered_results = filtered_results[(filtered_results.pval >= pval)]

	print(filtered_results[['random_seed', 'alpha', 'sigma', 'pred_loss', 'validation_pred_loss', 'pval']])
	filtered_results.drop(['pval'], inplace=True, axis=1)
	filtered_results.reset_index(drop=True, inplace=True)

	unique_filtered_results = filtered_results[['random_seed', 'sigma', 'alpha']].copy()
	unique_filtered_results.drop_duplicates(inplace=True)

	print("===valid hparams=====")
	print(unique_filtered_results.sort_values(['random_seed', 'sigma', 'alpha']))

	# get remaining configs
	# first_available_configs = pd.DataFrame(available_configs)
	# first_available_configs = first_available_configs.merge(filtered_results[['random_seed', 'alpha', 'sigma']],
	# 	on=['random_seed', 'alpha', 'sigma'])

	# first_available_configs = first_available_configs.to_dict(orient='records')

	# final_results = get_optimal_model_two_step(first_available_configs,
	# 	base_dir, hparams, weighted_xv, pval)


	return get_optimal_model_mmd(None, filtered_results, base_dir, hparams)



def get_optimal_model_classic(configs, filtered_results, base_dir, hparams):
	if ((configs is None) and (filtered_results is None)):
		raise ValueError("Need either configs or table of results_dict")

	if configs is not None:
		print("getting results")
		all_results, _ = import_results(configs, base_dir)
	else:
		all_results = filtered_results.copy()

	# print(all_results.alpha.value_counts())
	# assert 1==2 
	# ---get optimal hyperparams based on prediction loss
	# all_results['validation_pred_loss'] = all_results['validation_auc']
	#  print(all_results[['alpha', 'sigma', 'validation_pred_loss', 'validation_mmd']][(all_results.random_seed==22)])

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

	print(all_results[['random_seed', 'sigma', 'alpha', 'l2_penalty']])
	# TODO clean this up

	optimal_config_cols = ['random_seed', 'pflip0', 'pflip1', 'py0',
		'py1_y0', 'pixel', 'l2_penalty', 'dropout_rate', 'embedding_dim',
		'sigma', 'alpha', 'architecture', 'batch_size', 'weighted_mmd',
		'balanced_weights', 'minimize_logits', 'clean_back']
	if 'warmstart_dir' in list(all_results.columns):
		optimal_config_cols = optimal_config_cols + ['warmstart_dir']
	if 'two_way_mmd' in list(all_results.columns):
		optimal_config_cols = optimal_config_cols + ['two_way_mmd']

	optimal_configs = all_results.copy()
	optimal_configs = optimal_configs[optimal_config_cols]
	optimal_configs = optimal_configs.to_dict('records')

	# optimal_configs = [
	# 	collections.OrderedDict(config_dict) for config_dict in optimal_configs
	# ]

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

	return final_results_clean, optimal_configs


def get_optimal_model_mmd(configs, filtered_results, base_dir, hparams):
	if ((configs is None) and (filtered_results is None)):
		raise ValueError("Need either configs or table of results_dict")

	if configs is not None:
		all_results, _ = import_results(configs, base_dir)
	else:
		all_results = filtered_results.copy()

	# ---get optimal hyperparams based on prediction loss
	columns_to_keep = hparams + ['random_seed', 'validation_mmd']
	best_mmd = all_results[columns_to_keep]
	best_mmd = best_mmd.groupby('random_seed').validation_mmd.min()
	best_mmd = best_mmd.to_frame()

	best_mmd.reset_index(drop=False, inplace=True)
	best_mmd.rename(columns={'validation_mmd': 'min_validation_mmd'},
		inplace=True)
	all_results = all_results.merge(best_mmd, on='random_seed')
	all_results = all_results[
		(all_results.validation_mmd == all_results.min_validation_mmd)
	]

	print(all_results[['random_seed', 'sigma', 'alpha']].sort_values(['random_seed']))
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


