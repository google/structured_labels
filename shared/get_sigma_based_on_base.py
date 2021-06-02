import os
import functools
from waterbirds import data_builder
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats


from waterbirds import configurator
from waterbirds.data_builder import map_to_image_label, load_created_data
import shared.utils as utils
from shared import train_utils
from shared import evaluation_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

SIGMA_LIST = [1, 1e1, 1e2, 1e3]


def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	latest_model_dir = str(sorted(subdirs)[-1])
	loaded = tf.saved_model.load(latest_model_dir)
	model = loaded.signatures["serving_default"]
	return model

if __name__ == "__main__":
	experiment_name = "5050"

	p_tr = 0.8
	if experiment_name == '5050':
		py0 = 0.5
		py1_y0 = 0.5
	else:
		py0 = 0.8
		py1_y0 = 0.9

	py1_y0_shift_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .1]
	pflip0 = pflip1 = 0.01
	Kfolds = 5
	oracle_prop = 0.0
	random_seed = 0
	pixel = 128


	model_to_tune = "simple_baseline"
	oracle_prop = 0.0

	# --- get data
	experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
		f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}')

	_, valid_data, _ = load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=py1_y0_shift_list)
	map_to_image_label_given_pixel = functools.partial(map_to_image_label,
		pixel=pixel)

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = 64
	valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)

	# ---- loop through models
	all_config = configurator.get_sweep(experiment_name, model_to_tune, oracle_prop)

	all_results = []
	for config in all_config:
		if config['l2_penalty'] > 0.0:
			continue
		hash_string = utils.config_hasher(config)
		hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string, 'saved_model')

		model = get_last_saved_model(hash_dir)

		params = {
			'weighted_mmd': config['weighted_mmd'],
			'balanced_weights': config['balanced_weights'],
			'minimize_logits': config['minimize_logits'],
			'label_ind': 0}

		mmd_values = []
		for batch_id, examples in enumerate(valid_dataset):
			x, labels_weights = examples
			sample_weights, sample_weights_pos, sample_weights_neg = train_utils.extract_weights(
				labels_weights, params)

			labels = tf.identity(labels_weights['labels'])



			logits = model(tf.convert_to_tensor(x))['logits']
			zpred = model(tf.convert_to_tensor(x))['embedding']

			mmd_dict = evaluation_metrics.get_mmd_at_sigmas(SIGMA_LIST, labels, logits,
				zpred, sample_weights, sample_weights_pos, sample_weights_neg, params, True)

			mmd_values.append(pd.DataFrame(mmd_dict, index=[batch_id]))

		mmd_values = pd.concat(mmd_values, axis=0)
		mmd_mean = mmd_values.mean(axis=0)
		mmd_std = mmd_values.std(axis=0)
		print(mmd_mean)
		print(mmd_std)

		mmd_ratio = mmd_mean / mmd_std
		print(mmd_ratio)
		mmd_ratio = mmd_ratio.to_frame()
		mmd_ratio.reset_index(inplace=True, drop=False)

		mmd_ratio['sigma'] = mmd_ratio['index'].str.extract(r'(\d+.\d+)').astype('float')
		mmd_ratio.fillna(1.0, inplace=True)
		mmd_ratio.columns = ['col_val', 'ratio', 'sigma']
		opt_sigma = mmd_ratio['sigma'][(mmd_ratio.ratio == mmd_ratio.ratio.max())].values[0]

		curr_results = pd.DataFrame({
			'random_seed': config['random_seed'],
			'sigma': opt_sigma,
		}, index=[0])

		all_results.append(curr_results)

	all_results = pd.concat(all_results, axis=0, ignore_index=True)
	print(all_results)
	if model_to_tune == "simple_baseline":
		our_model_name = 'unweighted_slabs'
	else:
		our_model_name = 'slabs'
	all_results.to_csv(
		f'/data/ddmg/slabs/waterbirds/final_models/sigma_using_base_{our_model_name}_{experiment_name}.csv')
