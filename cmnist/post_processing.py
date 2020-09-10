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
import logging
import multiprocessing.pool
import os
import pickle

from absl import app
from absl import flags
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

from structured_labels import configurator

FLAGS = flags.FLAGS
flags.DEFINE_enum('exp_name', 'cmnist', ['cmnist', 'cmnist_no_overlap'],
									'Name of the experiment.')

flags.DEFINE_string('results_output_directory',
										'/usr/local/tmp/slabs/correlation_sweep/results',
										'Directory where the results should be saved')

MODELS = ['slabs', 'simple_baseline']
NUM_WORKERS = 10
DIRECTORY_TO_MODEL_MAPPER = {
	'/usr/local/tmp/slabs/correlation_sweep/cmnist_slabs':
		'slabs',
	'/usr/local/tmp/slabs/correlation_sweep/cmnist_simple_baseline':
		'simple_baseline',
	'/usr/local/tmp/slabs/correlation_sweep/cmnist_no_overlap_slabs':
		'slabs',
	'/usr/local/tmp/slabs/correlation_sweep/cmnist_no_overlap_simple_baselines':
		'simple_baseline'
}

GREEN = '#2ca02c'
RED = '#d62728'


def plot_errorbars_same_and_shifted(axis, model_results, metric, color):
	"""Plots results for same and shifted test distributions.

	Args:
		axis: matplotlib plot axis
		model_results: pandas dataframe with results for a given model. Must have
			mean and std for the same and shifted distributions for the required
			metric
		metric: metric to plot, one of loss or acc
		color: color for plotting

	Returns:
		None. Just adds the errorbars to an existing plot.
	"""
	axis.errorbar(
		model_results.py1_y0_s,
		model_results[f'{metric}_shift_mean'],
		yerr=model_results[f'{metric}_shift_std'],
		color=color)

	axis.errorbar(
		model_results.py1_y0_s,
		model_results[f'{metric}_same_mean'],
		yerr=model_results[f'{metric}_same_std'],
		color=color,
		linestyle='--')


def import_helper(config):
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
	exp_dir = config['exp_dir']
	random_seed = config['random_seed']
	py1_y0_s = config['py1_y0_s']
	alpha = config['alpha']
	sigma = config['sigma']
	l2_penalty = config['l2_penalty']
	dropout_rate = config['dropout_rate']
	dim = config['embedding_dim']

	mmd_param_str = f'alpha{alpha}_sigma{sigma}'
	regularization_param_str = f'l2{l2_penalty}_dropout{dropout_rate}_dim{dim}'
	experiment_param_str = f'pshift{py1_y0_s}_seed{int(random_seed)}'

	experiment_file = os.path.join(
		exp_dir,
		f'{mmd_param_str}_{regularization_param_str}_{experiment_param_str}',
		'performance_dump.pkl')

	if not os.path.exists(experiment_file):
		logging.error('Couldnt find %s', experiment_file)
		return None

	results_dict = pickle.load(open(experiment_file, 'rb'))
	model = DIRECTORY_TO_MODEL_MAPPER[exp_dir]
	return pd.DataFrame(
		{
			'model': model,
			'random_seed': random_seed,
			'py1_y0_s': py1_y0_s,
			'sigma': sigma,
			'alpha': alpha,
			'l2': l2_penalty,
			'embedding_dim': dim,
			'dropout': dropout_rate,
			'acc_valid': results_dict['validation']['accuracy'],
			'acc_same': results_dict['same_distribution']['accuracy'],
			'acc_shift': results_dict['shift_distribution']['accuracy'],
			'loss_valid': results_dict['validation']['loss'],
			'loss_same': results_dict['same_distribution']['loss'],
			'loss_shift': results_dict['shift_distribution']['loss']
		},
		index=[0])


def main(argv):
	del argv

	parameters = [
		configurator.get_sweep(FLAGS.exp_name, model) for model in MODELS
	]
	parameter_list = list(parameters)

	pool = multiprocessing.pool.ThreadPool(NUM_WORKERS)
	all_results = pool.map(import_helper, parameter_list)
	res = pd.concat(all_results, axis=0, ignore_index=True, sort=False)

	res.to_csv(
		os.path.join(FLAGS.results_output_directory,
			f'{FLAGS.exp_name}_all_results.csv'),
		index=False)
	res = res.groupby(
		['model', 'py1_y0_s', 'sigma', 'alpha', 'l2', 'embedding_dim',
		'dropout']).agg({'acc_valid': ['mean', 'std'],
			'acc_same': ['mean', 'std'],
			'acc_shift': ['mean', 'std'],
			'loss_valid': ['mean', 'std'],
			'loss_same': ['mean', 'std'],
			'loss_shift': ['mean', 'std']
		}).reset_index()
	res.columns = ['_'.join(col).strip() for col in res.columns.values]
	res.rename(
		{
			'model_': 'model',
			'py1_y0_s_': 'py1_y0_s',
			'sigma_': 'sigma',
			'alpha_': 'alpha',
			'l2_': 'l2',
			'dropout_': 'dropout',
			'embedding_dim_': 'embedding_dim',
		},
		axis=1,
		inplace=True)

	idx = res.groupby(
		['model',
		'py1_y0_s'])['loss_valid_mean'].transform(min) == res['loss_valid_mean']
	res_min_loss = res[idx].copy().reset_index(drop=True)
	res_slabs = res_min_loss[(res_min_loss.model == 'slabs')]
	res_simple_baseline = res_min_loss[(res_min_loss.model == 'simple_baseline')]

	_, axes = plt.subplots(1, 2, figsize=(14, 5))
	legend_elements = [
		Line2D([0], [0],
			color='black',
			lw=3,
			linestyle='--',
			label='Same distribution'),
		Line2D([0], [0], color='black', lw=3, label='Shifted distribution'),
		Patch(facecolor=RED, label='Ours'),
		Patch(facecolor=GREEN, label='Baseline')
	]

	plot_errorbars_same_and_shifted(axes[0], res_slabs, 'acc', RED)
	plot_errorbars_same_and_shifted(axes[0], res_simple_baseline, 'acc', GREEN)

	plot_errorbars_same_and_shifted(axes[1], res_slabs, 'loss', RED)
	plot_errorbars_same_and_shifted(axes[1], res_simple_baseline, 'loss', GREEN)

	axes[0].set_xlabel('Conditional probability in shifted distribution')
	axes[0].set_ylabel('Accuracy')
	axes[0].legend(handles=legend_elements, loc='lower right')

	axes[1].set_xlabel('Conditional probability in shifted distribution')
	axes[1].set_ylabel('Loss')
	axes[1].legend(handles=legend_elements, loc='upper right')
	plt.savefig(
		os.path.join(FLAGS.results_output_directory,
			f'{FLAGS.exp_name}_final_result.pdf'))
	plt.clf()
	plt.close()


if __name__ == '__main__':
	app.run(main)
