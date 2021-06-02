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
import os
import functools
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
from shared import train_utils
import multiprocessing
import tqdm
import collections
tf.autograph.set_verbosity(0)
pd.options.display.max_rows = 999


from waterbirds.data_builder import map_to_image_label, load_asymmetric_test_data
import shared.utils as utils
from shared import evaluation_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))
SHIFT_LIST = [.1]


def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	try:
		latest_model_dir = str(sorted(subdirs)[-1])
		loaded = tf.saved_model.load(latest_model_dir)
		model = loaded.signatures["serving_default"]
	except:
		print(estimator_dir)
		assert 1==2
	return model


def get_data(random_seed, clean_back, py0, py1_y0, pixel, pflip0):
	if clean_back == 'False':
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')
	else:
		experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
			f'cleanback_rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')

	shifted_data_dict = load_asymmetric_test_data(
		experiment_directory=experiment_directory, py1_y0_s=SHIFT_LIST)
	map_to_image_label_given_pixel = functools.partial(map_to_image_label,
		pixel=pixel)

	dataset_dict = {}
	for py in SHIFT_LIST:
		batch_size = len(shifted_data_dict[py])
		shifted_test_data = shifted_data_dict[py]
		eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
		eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
		eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
		dataset_dict[py] = iter(eval_shift_dataset)

	return dataset_dict


def get_risks(config):

	# -- get the hash directory where the model lives
	hash_string = utils.config_hasher(collections.OrderedDict(sorted(config.items())))
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string, 'saved_model')

	# -- get model
	model = get_last_saved_model(hash_dir)

	# -- get data for the model
	dataset_dict = get_data(config['random_seed'],
		config['clean_back'], config['py0'], config['py1_y0'],
		config['pixel'], config['pflip0'])

	metric_values = []
	for py in SHIFT_LIST:
		# print(f'-----P {py}----------')
		x, labels_weights = next(dataset_dict[py])
		logits = model(tf.convert_to_tensor(x))['logits']

		labels = tf.identity(labels_weights['labels'])
		y_main = tf.expand_dims(labels[:, 0], axis=-1)

		individual_losses = tf.keras.losses.binary_crossentropy(
			y_main, logits, from_logits=True)

		individual_losses = individual_losses.numpy()
		y_main = y_main.numpy()
		y_auxiliary = tf.expand_dims(labels[:, 1], axis=-1).numpy()

		# -- create masks
		y11 = y_auxiliary * y_main
		y01 = (1 - y_auxiliary) * y_main
		y10 = y_auxiliary * (1 - y_main)
		y00 = (1 - y_auxiliary) * (1 - y_main)

		R11 = np.mean(individual_losses[y11[:, 0] == 1])
		R01 = np.mean(individual_losses[y01[:, 0] == 1])
		R10 = np.mean(individual_losses[y10[:, 0] == 1])
		R00 = np.mean(individual_losses[y00[:, 0] == 1])

		# multiply by py
		py0 = np.mean(1 - y_main[: ,0]).squeeze()
		py1 = np.mean(y_main[:, 0]).squeeze()
		# py0 = 0.76
		# py1 = 1.0 - 0.76

		R_vec = np.array([[py0 * R00 - py0 * R10], [py1 * R01 - py1 * R11]])
		print("R_vec")
		print(R_vec)

		pv0_y0 = np.sum(y00[:, 0]) / np.sum((1 - y_main[:, 0]))
		pv0_y1 = np.sum(y01[:, 0]) / np.sum((y_main[:, 0]))

		# pv0_y0 = 1.0 - (0.5 * (1.0 - py))
		# pv0_y1 = 1.0 - py

		pv0 = 0.5
		P_vec = np.array([[pv0_y0 - pv0], [pv0_y1 - pv0]])
		print("P_vec")
		print(P_vec)

		proj_onto1 = np.full((2, 2), 0.5)

		denom = np.dot(R_vec.transpose(), P_vec) # 1 x 1
		num = np.dot(proj_onto1, P_vec)
		num = np.dot(R_vec.transpose(), num)
		ratio = num / denom
		ratio = ratio.squeeze()
		metric_values.append(pd.DataFrame({'random_seed': config['random_seed'],
			'py1_y0': py, 'ratio': ratio, 'denom': denom.squeeze(), 'num': num.squeeze(),
			'R': np.sum(R_vec).squeeze()}, index=[0]))

	metric_values = pd.concat(metric_values, ignore_index=True)
	return metric_values

def get_model_risks(optimal_configs):
	all_results = []
	# for config in optimal_configs:
	# 	res = get_risks(config)
	# 	results.append(res)


	pool = multiprocessing.Pool(20)
	for results in tqdm.tqdm(pool.imap_unordered(get_risks, optimal_configs),
		total=len(optimal_configs)):
		all_results.append(results)
	all_results = pd.concat(all_results, ignore_index=True)
	print(all_results)
	return all_results[['py1_y0', 'ratio', 'denom', 'num', 'R']].groupby('py1_y0').agg(["mean", "var"])

