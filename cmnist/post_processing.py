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
"""Post processing results from the cmnist experiments.

Script collects results from different experiment settings and different models
then produces the main plot.
"""

import itertools
import logging
import multiprocessing
import os
import pickle

from absl import app
from absl import flags
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from shared.utils import config_hasher, tried_config
from cmnist import configurator


FLAGS = flags.FLAGS
flags.DEFINE_enum('exp_name', 'correlation', ['correlation', 'overlap'],
									'Name of the experiment.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))
MODELS = ['slabs', 'opslabs', 'simple_baseline']
NUM_WORKERS = 10
X_AXIS_VAR = 'py1_y0_s'

MODEL_TO_PLOT_SPECS = {
	'slabs': {'color': '#ff7f0e', 'label': 'SLABS (ours)'},
	'opslabs': {'color': '#d62728', 'label': 'OP-SLABS (ours)'},
	'simple_baseline': {'color': '#2ca02c', 'label': 'Simple baseline'},
}


def plot_errorbars_same_and_shifted(axis,
																		legend_elements,
																		results,
																		model,
																		metric):
	"""Plots results for same and shifted test distributions.

	Args:
		axis: matplotlib plot axis
		legend_elements: list of legend elements to append to if
			None, no legend key is added for this model
		results: pandas dataframe with all models' results
		model: model to plot
		metric: metric to plot, one of loss or acc

	Returns:
		None. Just adds the errorbars to an existing plot.
	"""
	# TODO x-axis variable
	model_results = results[(results.model == model)]
	axis.errorbar(
		model_results.py1_y0_s,
		model_results[f'shift_distribution_{metric}_mean'],
		yerr=model_results[f'shift_distribution_{metric}_std'],
		color=MODEL_TO_PLOT_SPECS[model]['color'])

	axis.errorbar(
		model_results.py1_y0_s,
		model_results[f'same_distribution_{metric}_mean'],
		yerr=model_results[f'same_distribution_{metric}_std'],
		color=MODEL_TO_PLOT_SPECS[model]['color'],
		linestyle='--')
	model_legend_entry = Patch(facecolor=MODEL_TO_PLOT_SPECS[model]['color'],
		label=MODEL_TO_PLOT_SPECS[model]['label'])
	if legend_elements is not None:
		legend_elements.append(model_legend_entry)


def import_helper(args):
	"""Imports the dictionary with the results of an experiment.

	Args:
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
	# TODO update docstring
	model, config = args
	hash_string = config_hasher(config)
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')

	if not os.path.exists(performance_file):
		logging.error('Couldnt find %s', performance_file)
		return None

	results_dict = pickle.load(open(performance_file, 'rb'))
	results_dict.update(config)
	results_dict['model'] = model
	return pd.DataFrame(results_dict, index=[0])


def main(argv):
	del argv
	all_config = []
	for model in MODELS:
		model_configs = configurator.get_sweep(FLAGS.exp_name, model)
		available_configs = [tried_config(config, base_dir=BASE_DIR) for config
														in model_configs]
		model_configs = list(itertools.compress(model_configs, available_configs))
		all_config.extend([(model, config) for config in model_configs])

	pool = multiprocessing.Pool(NUM_WORKERS)
	res = []
	for config_res in tqdm.tqdm(pool.imap_unordered(import_helper, all_config),
		total=len(all_config)):
		res.append(config_res)

	res = pd.concat(res, axis=0, ignore_index=True, sort=False)

	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	res.to_csv(
		os.path.join(results_dir, f'{FLAGS.exp_name}_xval_results.csv'),
		index=False)
	# TODO: x-axis variable
	res = res.groupby(
		['model', 'py1_y0_s', 'sigma', 'alpha', 'l2_penalty', 'embedding_dim',
		'dropout_rate']).agg({
			'validation_accuracy': ['mean', 'std'],
			'same_distribution_accuracy': ['mean', 'std'],
			'shift_distribution_accuracy': ['mean', 'std'],
			'validation_loss': ['mean', 'std'],
			'same_distribution_loss': ['mean', 'std'],
			'shift_distribution_loss': ['mean', 'std']
		}).reset_index()
	res.columns = ['_'.join(col).strip() for col in res.columns.values]
	res.rename(
		{
			'model_': 'model',
			'py1_y0_s_': 'py1_y0_s',
			'sigma_': 'sigma',
			'alpha_': 'alpha',
			'l2_penalty_': 'l2_penalty',
			'dropout_rate_': 'dropout_rate',
			'embedding_dim_': 'embedding_dim',
		},
		axis=1,
		inplace=True)

	idx = res.groupby(
		['model',
		X_AXIS_VAR])['validation_loss_mean'].transform(min) == res[
		'validation_loss_mean']
	res_min_loss = res[idx].copy().reset_index(drop=True)

	_, axes = plt.subplots(1, 2, figsize=(14, 5))
	legend_elements = [
		Line2D([0], [0],
			color='black',
			lw=3,
			linestyle='--',
			label='Same distribution'),
		Line2D([0], [0], color='black', lw=3, label='Shifted distribution')
	]

	for model in MODELS:
		plot_errorbars_same_and_shifted(axes[0], legend_elements, res_min_loss,
			model, 'accuracy')
		plot_errorbars_same_and_shifted(axes[1], None, res_min_loss,
			model, 'loss')

	axes[0].set_xlabel('Conditional probability in shifted distribution')
	axes[0].set_ylabel('Accuracy')
	axes[0].legend(handles=legend_elements, loc='lower right')

	axes[1].set_xlabel('Conditional probability in shifted distribution')
	axes[1].set_ylabel('Loss')
	axes[1].legend(handles=legend_elements, loc='upper right')
	plt.savefig(os.path.join(results_dir, f'{FLAGS.exp_name}_plot.pdf'))
	plt.clf()
	plt.close()


if __name__ == '__main__':
	app.run(main)
