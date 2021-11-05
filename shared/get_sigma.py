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

"""Creates config dictionaries for different experiments and models waterbirds"""
import os
import functools
from pathlib import Path
from random import sample
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
from shared import train_utils
import multiprocessing
import tqdm
# import random
tf.autograph.set_verbosity(0)



import waterbirds.data_builder as wb
import chexpert.data_builder as chx
import cmnist.data_builder as cm

import shared.utils as utils
from shared import evaluation_metrics


def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	try:
		latest_model_dir = str(sorted(subdirs)[-1])
		loaded = tf.saved_model.load(latest_model_dir)
		model = loaded.signatures["serving_default"]
	except:
		print(estimator_dir)
	return model


def get_data_waterbirds(kfolds, random_seed, clean_back, py0, py1_y0, pixel, pflip0):
	if clean_back == 'False':
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')
	else:
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'cleanback_rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')

	_, valid_data, _, _ = wb.load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=[.5])
	map_to_image_label_given_pixel = functools.partial(wb.map_to_image_label,
		pixel=pixel)

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = int(len(valid_data) / kfolds)
	valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
	return valid_dataset


def get_data_chexpert(kfolds, random_seed, skew_train, pixel):

	experiment_directory = (f'/data/ddmg/slabs/chexpert/experiment_data/rs{random_seed}')

	_, valid_data, _ = chx.load_created_data(
		experiment_directory=experiment_directory, skew_train=skew_train)
	map_to_image_label_given_pixel = functools.partial(chx.map_to_image_label,
		pixel=pixel)
	# to avoid running running into OOM issues
	if len(valid_data) > 1000:
		valid_data = sample(valid_data, 1000)

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = int(len(valid_data) / kfolds)
	valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
	return valid_dataset


def get_optimal_sigma_for_run(config, kfolds, weighted_xv):
	# -- get the dataset
	# print("get data")
	if 'skew_train' in config.keys():
		valid_dataset = get_data_chexpert(kfolds, config['random_seed'], config['skew_train'],
			config['pixel'])
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
			'chexpert'))

	elif 'clean_back' in config.keys():
		valid_dataset = get_data_waterbirds(kfolds, config['random_seed'], config['clean_back'],
			config['py0'], config['py1_y0'], config['pixel'], config['pflip0'])
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
			'waterbirds'))

	# -- get the hash directory where the model lives
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')

	# -- get model
	model = get_last_saved_model(hash_dir)

	# -- set parameters for calculating the mmd
	params = {
		'weighted_mmd': config['weighted_mmd'],
		'balanced_weights': config['balanced_weights'],
		'minimize_logits': config['minimize_logits'],
		'sigma': config['sigma'],
		'alpha': config['alpha'],
		'label_ind': 0}

	if weighted_xv == 'weighted':
		params['weighted_mmd'] = 'True'
	if weighted_xv == 'weighted_bal':
		params['weighted_mmd'] = 'True'
		params['balanced_weights'] = 'True'

	metric_values = []
	for batch_id, examples in enumerate(valid_dataset):
		# print(f'{batch_id} / {kfolds}')
		x, labels_weights = examples
		sample_weights, sample_weights_pos, sample_weights_neg = train_utils.extract_weights(
			labels_weights, params)
		labels = tf.identity(labels_weights['labels'])

		logits = model(tf.convert_to_tensor(x))['logits']
		zpred = model(tf.convert_to_tensor(x))['embedding']


		metric_value = evaluation_metrics.get_mmd_at_sigmas([config['sigma']], labels, logits,
			zpred, sample_weights, sample_weights_pos, sample_weights_neg, params, True)
		metric_value = list(metric_value.values())[0]

		metric_values.append(metric_value)

	curr_results = pd.DataFrame({
		'random_seed': config['random_seed'],
		'alpha': config['alpha'],
		'sigma': config['sigma'],
		'mmd': np.mean(metric_values),
		'pval': stats.ttest_1samp(metric_values, 0.0)[1]
	}, index=[0])
	if (np.mean(metric_values) == 0.0 and np.var(metric_values) == 0.0):
		curr_results['pval'] = 1
	return curr_results


def get_diff_to_best_pred_loss(x, kfolds):
	fold_pred_loss_cols = [f'pred_loss_{fold_id}' for fold_id in range(kfolds)]
	metric_values = x.loc[fold_pred_loss_cols].tolist()

	fold_best_loss_cols = [f'best_{col}' for col in fold_pred_loss_cols]
	best_loss = x.loc[fold_best_loss_cols]

	return stats.ttest_ind(metric_values, best_loss)[1]


def get_optimal_sigma(all_config, kfolds, weighted_xv):
	all_results = []
	runner_wrapper = functools.partial(get_optimal_sigma_for_run, kfolds=kfolds,
		weighted_xv=weighted_xv)

	pool = multiprocessing.Pool(20)
	for results in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config), total=len(all_config)):
		all_results.append(results)

	all_results = pd.concat(all_results, axis=0, ignore_index=True)
	return all_results





