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

"""Gets predictions from the optimal model for each replicate"""
import os
import functools
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd

import multiprocessing
import tqdm
import pickle
from sklearn.metrics import roc_auc_score
import argparse


import waterbirds.data_builder as wb
import chexpert.data_builder as chx

# import random
tf.autograph.set_verbosity(0)


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


def get_data_waterbirds(py1_y0_s, random_seed, clean_back, py0, py1_y0, pixel,
	pflip0):
	if clean_back == 'False':
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')
	else:
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'cleanback_rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')

	_, _, test_data_dict, _ = wb.load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=[py1_y0_s])
	test_data = test_data_dict[py1_y0_s]

	map_to_image_label_given_pixel = functools.partial(wb.map_to_image_label,
		pixel=pixel)

	test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
	test_dataset = test_dataset.map(map_to_image_label_given_pixel,
		num_parallel_calls=1)
	batch_size = int(len(test_data))
	test_dataset = test_dataset.batch(batch_size).repeat(1)
	return test_dataset


def get_data_chexpert(py1_y0_s, random_seed, skew_train, pixel):

	experiment_directory = ('/data/ddmg/slabs/chexpert/experiment_data/'
		f'rs{random_seed}')

	_, _, test_data_dict = chx.load_created_data(
		experiment_directory=experiment_directory, skew_train=skew_train)
	test_data = test_data_dict[py1_y0_s]

	map_to_image_label_given_pixel = functools.partial(chx.map_to_image_label,
		pixel=pixel)

	test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
	test_dataset = test_dataset.map(map_to_image_label_given_pixel,
		num_parallel_calls=1)
	batch_size = 500
	test_dataset = test_dataset.batch(batch_size).repeat(1)
	return test_dataset


def get_predictions(rs_config, py1_y0_s, base_dir):
	# -- get the optimal hash and current random seed
	random_seed = rs_config['random_seed']
	hash_string = rs_config['hash']
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')

	config_file = os.path.join(base_dir, 'tuning', hash_string, 'config.pkl')
	config = pickle.load(open(config_file, "rb"))

	# -- get the dataset
	# print("get data")
	if 'skew_train' in config.keys():
		test_dataset = get_data_chexpert(py1_y0_s, config['random_seed'],
			config['skew_train'], config['pixel'])
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
			'..','chexpert'))

	elif 'clean_back' in config.keys():
		test_dataset = get_data_waterbirds(py1_y0_s, config['random_seed'],
			config['clean_back'], config['py0'], config['py1_y0'], config['pixel'],
			config['pflip0'])
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
			'waterbirds'))
	else:
		raise ValueError("not implemented")
	# -- get model
	model = get_last_saved_model(hash_dir)

	pred_df_list = []
	for batch_id, examples in enumerate(test_dataset):
		# print(f'{batch_id}')
		x, labels_weights = examples
		predictions = model(tf.convert_to_tensor(x))['probabilities']

		pred_df = pd.DataFrame(labels_weights['labels'].numpy())
		pred_df.columns = ['y', 'v']
		pred_df['predictions'] = predictions.numpy()
		pred_df['pred_class'] = (pred_df.predictions >= 0.5)*1.0

		pred_df_list.append(pred_df)

		# print(f"----{random_seed}----")
		# print(pred_df[(pred_df.v ==0)]['y'].value_counts(normalize=True))
		# print(pred_df[(pred_df.v ==1)]['y'].value_counts(normalize=True))

	pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)


	# TODO this assumes we will always just care about DNNS. 
	pred_df.to_csv(f'{base_dir}/final_models/predictions{py1_y0_s}_DNN_rs{random_seed}.csv', index=False)
	return None 


def main(model, py1_y0_s, dataset, experiment_name, xv_method, pval, batch_size,
	clean_back):

	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
		dataset))
	final_models_dir = f'{base_dir}/final_models'

	if xv_method != 'classic':
		xv_method = f'{xv_method}{pval}'

	# get the list of configs
	if dataset == 'waterbirds':
		optimal_configs = pd.read_csv(
			(f'{final_models_dir}/optimal_config_{model}_{xv_method}_{experiment_name}'
				f'_{clean_back}_{batch_size}.csv')
		)
	elif dataset == 'chexpert':
		optimal_configs = pd.read_csv(
			(f'{final_models_dir}/optimal_config_{model}_{xv_method}_{experiment_name}'
				'.csv')
		)

		del batch_size, clean_back

	all_config = [
		optimal_configs.iloc[i] for i in range(optimal_configs.shape[0])
	]

	all_results = []
	runner_wrapper = functools.partial(get_predictions, py1_y0_s=py1_y0_s,
		base_dir=base_dir)

	# for cid, config in enumerate(all_config):
	# 	print(cid)
	# 	results = runner_wrapper(config)
	# 	all_results.append(results)

	pool = multiprocessing.Pool(20)
	for _ in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
		total=len(all_config)):
		pass



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', '-dataset',
		default='waterbirds',
		choices=['waterbirds', 'chexpert'],
		help="Which task",
		type=str)

	parser.add_argument('--experiment_name', '-experiment_name',
		default='5050',
		choices=['5050', '5090', '8090', '8050', '8090_asym', 'skew_train'],
		help="Which experiment to run",
		type=str)

	parser.add_argument('--model', '-model',
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

	parser.add_argument('--xv_method', '-xv_method',
		default='classic',
		choices=['classic', 'ts', 'uts'],
		help="Which cross validation method",
		type=str)

	parser.add_argument('--batch_size', '-batch_size',
		default=64,
		help=("training batch size"),
		type=int)

	parser.add_argument('--clean_back', '-clean_back',
		default='True',
		help="Run with clean or noisy backgrounds",
		type=str)

	parser.add_argument('--py1_y0_s', '-py1_y0_s',
		default=0.1,
		help="P(Y|V)",
		type=float)

	parser.add_argument('--pval', '-pval',
		default=0.05,
		help="Pvalue for two step",
		type=float)

	args = vars(parser.parse_args())
	main(**args)
