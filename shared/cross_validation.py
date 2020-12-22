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


def get_final_results(results):
	shift_columns = [col for col in results.columns if 'shift' in col]

	shift_accuracy_loss_columns = [
		col for col in shift_columns if ('loss' in col) or ('accuracy' in col) or ('auc' in col)
	]
	results = results[shift_accuracy_loss_columns]
	results = results.agg({
		col: ['mean', 'std'] for col in shift_accuracy_loss_columns
	})
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

	results_loss = results[(results.index.str.contains('loss'))]
	results_loss = results_loss.rename(columns={
		col: f'loss_{col}' for col in results_loss.columns if col != 'py1_y0_s'
	})


	results_final = results_accuracy.merge(results_loss, on=['py1_y0_s'])
	results_final = results_final.merge(results_auc, on=['py1_y0_s'])
	return results_final


def get_optimal_model_results(mode, configs, base_dir, hparams, equivalent=True,
	pval=False):

	if mode not in ['classic', 'two_step']:
		raise NotImplementedError('Can only run classic or two_step modes')

	# if mode == 'two_step' and sorted(hparams) == ['alpha', 'sigma']:
	# 	optimal_model_results = get_optimal_model_two_step(configs, base_dir, pval)
	# 	return optimal_model_results

	if mode == "two_step":
		if sorted(hparams) != ['alpha', 'dropout_rate', 'embedding_dim', 'l2_penalty', 'sigma']:
			raise NotImplementedError('Have not implemented these hyperparams yet!')
		optimal_model_results = get_optimal_model_two_step(configs,
			base_dir, pval)
		return optimal_model_results

	optimal_model_results = get_optimal_model_classic(configs, base_dir, hparams)
	return optimal_model_results


def get_optimal_model_classic(configs, base_dir, hparams):
	res_all = import_results(configs, base_dir)

	# ---get optimal hyperparams
	columns_to_keep = hparams + ['validation_loss']
	res_hparams = res_all[columns_to_keep]
	res_hparams = res_hparams.groupby(hparams).agg({'validation_loss': ['mean']})
	res_hparams.reset_index(inplace=True)
	res_hparams.columns = [
		'_'.join(col).strip() for col in res_hparams.columns.values
	]
	res_hparams.rename({f'{hparam}_': hparam for hparam in hparams}, axis=1,
		inplace=True)
	best_loss = res_hparams.validation_loss_mean.min()
	res_hparams = res_hparams[(res_hparams.validation_loss_mean == best_loss)]
	res_hparams = res_hparams[hparams]
	print(res_hparams)
	res = res_all.merge(res_hparams, on=hparams)
	res_final = get_final_results(res)


	res_per_run_final = []
	for rs in res.random_seed.unique():
		res_rs = res[(res.random_seed == rs)]
		res_rs = get_final_results(res_rs)
		std_cols = [col for col in res_rs if 'std' in col]
		res_rs.drop(std_cols, axis=1, inplace=True)
		res_rs['random_seed'] = rs
		res_per_run_final.append(res_rs)
	res_per_run_final = pd.concat(res_per_run_final, axis=0).reset_index(drop=True)

	return res_final, res_per_run_final


def get_optimal_sigma(validation_data):
	columns_to_keep = ['alpha', 'sigma', 'validation_mmd']
	res_sigma = validation_data[columns_to_keep]
	res_sigma = res_sigma.groupby(['alpha', 'sigma']).agg(
		{'validation_mmd': ['mean']})
	res_sigma.reset_index(inplace=True)
	res_sigma.columns = [
		'_'.join(col).strip() for col in res_sigma.columns.values
	]
	res_sigma.rename(
		{f'{hparam}_': hparam for hparam in ['alpha', 'sigma']}, axis=1,
		inplace=True)

	# TODO: this assumes that each sigma has the max value of alpha
	# for arbitrarily large alphas, if mmd > 0, then sigma is not good
	res_sigma['min_mmd_for_sigma'] = res_sigma.validation_mmd_mean
	res_sigma['min_mmd_for_sigma'] = res_sigma.groupby('sigma')['min_mmd_for_sigma'].transform('min')

	min_validation_mmd = max(1e-3, np.min(res_sigma.validation_mmd_mean))
	res_sigma['zero_at_max_alpha'] = np.where(res_sigma['min_mmd_for_sigma'] <= min_validation_mmd,
		1, 0)
	res_sigma['zero_at_max_alpha'] = res_sigma.groupby('sigma')['zero_at_max_alpha'].transform('max')
	res_sigma = res_sigma[(res_sigma.zero_at_max_alpha == 1)].reset_index(drop=True)
	print(res_sigma.sort_values(['sigma', 'alpha']))
	res_sigma.drop(['zero_at_max_alpha', 'min_mmd_for_sigma'], axis=1,
		inplace=True)

	# TODO this needs to be done for each sigma separately
	# if res_sigma.alpha.min() == res_sigma.alpha.max():
		# print(res_sigma)
		# raise ValueError('Need at least 2 unique values of alpha for')
	res_sigma = res_sigma[((res_sigma.alpha == res_sigma.alpha.min()) | (
		res_sigma.alpha == res_sigma.alpha.max()))]
	res_sigma = res_sigma.groupby('sigma')['validation_mmd_mean'].agg(
		np.ptp).reset_index()
	min_validation_mmd = min(1e-3, np.max(res_sigma.validation_mmd_mean))
	print(min_validation_mmd)
	print(res_sigma)
	res_sigma = res_sigma[(res_sigma.validation_mmd_mean >= min_validation_mmd)]
	optimal_sigma = np.min(res_sigma.sigma)
	return optimal_sigma


def get_equivalent_hparams(validation_data, hparams, pval):
	if 'validation_loss_mean' not in validation_data.columns.tolist():
		columns_to_keep = hparams + ['validation_loss', 'validation_mmd']
		validation_data = validation_data[columns_to_keep]
		validation_data['count'] = 1

		validation_data = validation_data.groupby(hparams).agg(
			{
				'validation_loss': ['mean', 'std'],
				'validation_mmd': ['mean', 'std'],
				'count': ['sum']
			}).reset_index()
		validation_data.columns = [
			'_'.join(col).strip() for col in validation_data.columns.values
		]
		validation_data.rename({f'{hparam}_': hparam for hparam in hparams},
			axis=1, inplace=True)

	validation_data = validation_data.sort_values(
		'validation_loss_mean').reset_index()

	min_val_loss = validation_data.validation_loss_mean[0]
	min_val_count = validation_data.count_sum[0]
	if pval:
		min_val_loss_std = validation_data.validation_loss_std[0]
		pvals = [ttest(
			mean1=min_val_loss, std1=min_val_loss_std, nobs1=min_val_count,
			mean2=validation_data.validation_loss_mean[i],
			std2=validation_data.validation_loss_std[i],
			nobs2=validation_data.count_sum[i]
		).pvalue for i in range(validation_data.shape[0])]
		validation_data['pvals'] = pvals
		validation_data['min_validation_loss'] = np.where(
			validation_data.pvals > 0.05, 1, 0)
		validation_data.drop('pvals', axis=1, inplace=True)
	else:
		min_val_loss_ste = validation_data.validation_loss_std[0] / np.sqrt(
			min_val_count)
		validation_data['min_validation_loss'] = np.where(
			validation_data.validation_loss_mean <= min_val_loss + min_val_loss_ste,
			1, 0)
	if validation_data.min_validation_loss.max() == 0:
		validation_data.min_validation_loss.loc[0] = 1
	validation_data = validation_data[(validation_data.min_validation_loss == 1)]
	equivalent_hparams = validation_data[hparams + ['validation_mmd_mean', 'validation_mmd_std']]
	return equivalent_hparams


def get_optimal_model_two_step(configs, base_dir, pval):
	print("starting import")
	res_all = import_results(configs, base_dir)

	print("get sigma")
	# --- get optimal sigma
	optimal_sigma = get_optimal_sigma(res_all)

	# --- get optimal alpha
	print("get alpha")
	res_alpha = res_all[(res_all.sigma == optimal_sigma)].reset_index(drop=True)

	equivalent_hparams = get_equivalent_hparams(res_alpha, ['alpha', 'sigma'],
		pval)
	# equivalent_hparams = res_alpha
	optimal_alpha = equivalent_hparams.alpha.max()
	print("get final")
	# --- get the optimal model results
	res = res_all[((res_all.sigma == optimal_sigma) & (
		res_all.alpha == optimal_alpha))].reset_index(drop=True)
	print(optimal_sigma, optimal_alpha)
	res_final = get_final_results(res)

	# TODO: check if this is ok
	res_per_run_final = []
	for rs in res.random_seed.unique():
		res_rs = res[(res.random_seed == rs)]
		res_rs = get_final_results(res_rs)
		std_cols = [col for col in res_rs if 'std' in col]
		res_rs.drop(std_cols, axis=1, inplace=True)
		res_rs['random_seed'] = rs
		res_per_run_final.append(res_rs)
	res_per_run_final = pd.concat(res_per_run_final, axis=0).reset_index(drop=True)
	return res_final, res_per_run_final


def get_optimal_model_with_lambda_two_step_elaborate(configs, base_dir, pval):

	res_all = import_results(configs, base_dir)

	lambda_combinations = res_all[['dropout_rate', 'l2_penalty',
		'embedding_dim']].copy()
	lambda_combinations.drop_duplicates(inplace=True)
	lambda_combinations.reset_index(inplace=True, drop=True)

	best_alpha_sigmas = []
	for i in range(lambda_combinations.shape[0]):
		lambda_vals = lambda_combinations.iloc[[i]].copy()

		# -- get all the validation results for this lambda
		res_lambda = res_all.merge(lambda_vals, on=lambda_vals.columns.tolist())

		# -- get best sigma for this lambda
		sigma_lambda = get_optimal_sigma(res_lambda)

		# -- get best alpha for this lambda
		res_alpha_lambda = res_lambda[(
			res_lambda.sigma == sigma_lambda)].reset_index(drop=True)
		if res_alpha_lambda.shape[0] == 0:
			continue

		equivalent_alphas_lambda = get_equivalent_hparams(res_alpha_lambda,
			['alpha', 'sigma'], pval)
		alpha_lambda = equivalent_alphas_lambda.alpha.max()

		lambda_vals['sigma'] = sigma_lambda
		lambda_vals['alpha'] = alpha_lambda
		best_alpha_sigmas.append(lambda_vals)

	# -- get all the data for lambda, alpha_lambda, sigma_lambda
	best_alpha_sigmas = pd.concat(best_alpha_sigmas, axis=0)

	# --- get the equivalent models
	res_lambda_sigma_alpha = res_all.merge(best_alpha_sigmas,
		on=best_alpha_sigmas.columns.tolist()).reset_index()

	equivalent_lambda = get_equivalent_hparams(res_lambda_sigma_alpha,
			['alpha', 'sigma', 'dropout_rate', 'l2_penalty', 'embedding_dim'], pval)
	equivalent_lambda['ratio'] = equivalent_lambda.alpha / equivalent_lambda.sigma
	print(equivalent_lambda)
	best_lambda = equivalent_lambda[(
		equivalent_lambda.ratio == equivalent_lambda.ratio.max())]
	best_lambda = best_lambda[['dropout_rate', 'l2_penalty', 'embedding_dim',
		'alpha', 'sigma']].reset_index(drop=True)
	print(best_lambda)
	if best_lambda.shape[0] > 1:
		random_model = np.random.choice(best_lambda.shape[0], size=1).tolist()
		best_lambda = best_lambda.loc[random_model]
		print(best_lambda)

	res = res_all.merge(best_lambda, on=best_lambda.columns.tolist())
	res_final = get_final_results(res)
	return res_final


def get_optimal_model_with_lambda_two_step(configs, base_dir, equivalent, pval):

	hparams = ['alpha', 'sigma', 'dropout_rate', 'l2_penalty',
		'embedding_dim']
	res_all = import_results(configs, base_dir)

	validation_data = res_all[hparams + ['validation_mmd',
		'validation_loss']].copy()
	validation_data['count'] = 1
	validation_data = validation_data.groupby(hparams).agg(
		{
			'validation_loss': ['mean', 'std'],
			'validation_mmd': ['mean', 'std'],
			'count': ['sum']
		}).reset_index()


	validation_data.columns = [
		'_'.join(col).strip() for col in validation_data.columns.values
	]
	validation_data.rename({f'{hparam}_': hparam for hparam in hparams},
		axis=1, inplace=True)
	print(validation_data[['alpha', 'sigma', 'validation_loss_mean', 'validation_mmd_mean']])
	# validation_data = validation_data[(validation_data.alpha == 1e5)]
	# validation_data = validation_data[
	# ((validation_data.validation_mmd_mean - validation_data.validation_mmd_std) <= 0)]


	# pvals = [ttest(
	# 	mean1=0, std1=1e-5, nobs1=10,
	# 	mean2=validation_data.validation_mmd_mean[i],
	# 	std2=validation_data.validation_mmd_std[i],
	# 	nobs2=validation_data.count_sum[i]
	# ).pvalue for i in range(validation_data.shape[0])]
	# validation_data['pvals'] = pvals
	# validation_data = validation_data[(validation_data.pvals > 0.05)]
	# validation_data.drop('pvals', axis =1, inplace = True)

	validation_data.reset_index(drop=True, inplace=True)


	# print(validation_data[['alpha', 'sigma', 'embedding_dim', 'validation_mmd_mean']])

	# ---- IF EQUIVALENT
	# if equivalent

	equivalent_models = get_equivalent_hparams(validation_data,
		['alpha', 'sigma', 'dropout_rate', 'l2_penalty', 'embedding_dim'], pval=pval)
	# equivalent_models['ratio'] = equivalent_models.alpha / equivalent_models.sigma
	# print(equivalent_models)
	best_lambda = equivalent_models[(
		equivalent_models.validation_mmd_mean == equivalent_models.validation_mmd_mean.min())]

	# # ----IF NOT EQUIVALENT
	# else:
	# 	best_lambda = validation_data[
	# 		(validation_data.validation_loss_mean == validation_data.validation_loss_mean.min())]

	# ------
	best_lambda = best_lambda[['dropout_rate', 'l2_penalty', 'embedding_dim',
		'alpha', 'sigma']].reset_index(drop=True)

	print(best_lambda)
	if best_lambda.shape[0] > 1:
		best_lambda = best_lambda[(best_lambda.embedding_dim == best_lambda.embedding_dim.max())]
		best_lambda.reset_index(drop=True, inplace=True)

	if best_lambda.shape[0] > 1:
		random_model = np.random.choice(best_lambda.shape[0], size=1).tolist()
		best_lambda = best_lambda.loc[random_model]
		print(best_lambda)

	res = res_all.merge(best_lambda, on=best_lambda.columns.tolist())
	res_final = get_final_results(res)

	res_per_run_final = []
	for rs in res.random_seed.unique():
		res_rs = res[(res.random_seed == rs)]
		res_rs = get_final_results(res_rs)
		std_cols = [col for col in res_rs if 'std' in col]
		res_rs.drop(std_cols, axis=1, inplace=True)
		res_rs['random_seed'] = rs
		res_per_run_final.append(res_rs)
	res_per_run_final = pd.concat(res_per_run_final, axis=0).reset_index(drop=True)

	return res_final, res_per_run_final

