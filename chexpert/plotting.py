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


import os
from absl import app
from absl import flags
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfCropMargins import crop
plt.style.use('tableau-colorblind10')

FLAGS = flags.FLAGS
flags.DEFINE_enum('metric', 'auc', ['auc', 'accuracy'], 'group results by dataset or model?')
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_enum('groupby', 'dataset', ['dataset', 'model'], 'group results by dataset or model?')



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'chexpert'))


MIN_METRIC = 0.4

def get_model_dict(pval):

	model_specs = {
		f'slabs_weighted_bal_ts{pval}': {
			'label': 'wMMD-reg-T', 'linestyle': 'solid',
			'color': '#C85200'
		},
		# f'slabs_weighted_bal_classic': {
		# 	'label': 'wMMD-reg-C', 'linestyle': 'solid',
		# 	'color': '#C85200'
		# },
		'simple_baseline_classic': {
			'label': 'L2-reg-C', 'linestyle': 'dotted',
			'color': 'black'
		},
		'weighted_baseline_classic': {
			'label': 'wL2-reg-C', 'linestyle': 'solid',
			'color': '#ABABAB'
		},

		'random_aug_classic': {
			'label': 'Rand-Aug-C', 'linestyle': 'solid',
			'color': '#5F6B29'
		},

	}


	return model_specs


def plot_errorbars_dataset_groups(model_to_spec, axis,  results, model, metric, barwidth, modelid):
	"""Plots results for same and shifted test distributions.

	Args:
		axis: matplotlib plot axis
		results: pandas dataframe with all models' results
		model: model to plot
		metric: metric to plot, one of loss or acc

	Returns:
		None. Just adds the errorbars to an existing plot.
	"""
	# TODO x-axis variable
	model_results = results[(results.model == model)]
	model_results['x_pos'] = model_results['x_pos'] + modelid * barwidth
	axis.bar(
		x = model_results.x_pos,
		height=model_results[f'{metric}_mean'],
		width = barwidth, 
		yerr=model_results[f'{metric}_std'] / np.sqrt(5),
		color=model_to_spec[model]['color'],
		align='edge', 
		linewidth=3,
		label=model_to_spec[model]['label'],
		capsize=5)

def plot_errorbars_model_groups(axis,  results, model, metric, barwidth, modelid):
	"""Plots results for same and shifted test distributions.

	Args:
		axis: matplotlib plot axis
		results: pandas dataframe with all models' results
		model: model to plot
		metric: metric to plot, one of loss or acc

	Returns:
		None. Just adds the errorbars to an existing plot.
	"""
	# TODO x-axis variable
	model_results = results[(results.model == model)]
	model_results['x_pos'] = model_results['x_pos'] + modelid
	axis.bar(
		x = model_results.x_pos,
		height=model_results[f'{metric}_mean'],
		width = barwidth, 
		yerr=model_results[f'{metric}_std'] / np.sqrt(5),
		color=model_results['color'],
		align='edge', 
		linewidth=3,
		capsize=5)


def main(argv):
	del argv
	model_to_spec = get_model_dict(FLAGS.pval)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')


	# --- collect the results 
	res = []
	for experiment_name in ['skew_train', 'unskew_train']:
		for model in model_to_spec.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{experiment_name}.csv')
				# TODO: remove this part 
				model_res = model_res[((model_res.py1_y0_s == 0.5) | (model_res.py1_y0_s == 0.9))]
				model_res['test_skew'] = np.where(model_res.py1_y0_s == 0.5, 'test: unskew', 'test: skew')
				train_skew = 'train: skew' if experiment_name == 'skew_train' else 'train: unskew'
				model_res['train_test'] = train_skew + '\n' + model_res['test_skew']
			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)

	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	available_models = res.model.unique().tolist()
	
	# --- compile the plot 
	plt.figure(figsize=(7, 5))
	font = {'size': 16, 'family': 'serif', 'serif': 'Computer Modern Roman'}
	plt.rc('font', **font)
	plt.rc('text', usetex=True)

	if FLAGS.groupby == 'dataset':
		x_pos_mapping = {
			'train: skew\ntest: skew': 0, 
			'train: unskew\ntest: unskew': 1, 
			'train: skew\ntest: unskew': 2, 
			'train: unskew\ntest: skew': 3
			}

		res['x_pos'] = res['train_test'].map(x_pos_mapping)
		barwidth = 0.2  # the width of the bars

		
		for modelid, model in enumerate(available_models):
			print(f'plot {model}')
			plot_errorbars_dataset_groups(model_to_spec, plt, res, model, FLAGS.metric, barwidth, modelid)

		savename = os.path.join(results_dir,
			f'chexpert_{FLAGS.pval}_{FLAGS.metric}_{FLAGS.groupby}.pdf')
		cropped_savename = os.path.join(results_dir,
			f'chexpert_{FLAGS.pval}_{FLAGS.metric}_{FLAGS.groupby}_cropped.pdf')

		plt.ylim(bottom = MIN_METRIC)
		if FLAGS.metric == 'auc':
			plt.ylabel('AUROC')
		else:
			plt.ylabel('Accuracy')

		dataset_to_indic = {value:key for key, value in x_pos_mapping.items()}
		xlabels = [dataset_to_indic[i] for i in range(4)]

		midpoint = (barwidth * len(available_models))/2.0
		plt.xticks(ticks = [i + midpoint for i in range(4)], labels = xlabels)

		
		plt.legend(loc='upper right', prop={'size': 12})
		plt.tight_layout()
		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])

	elif FLAGS.groupby == 'model':
		barwidth = 0.2  # the width of the bars
		x_pos_mapping = {
			'train: skew\ntest: skew': barwidth*0, 
			'train: unskew\ntest: unskew': barwidth*2, 
			'train: skew\ntest: unskew': barwidth*1, 
			'train: unskew\ntest: skew': barwidth*3
			}

		res['x_pos'] = res['train_test'].map(x_pos_mapping)
		

		color_mapping = {
			'train: skew\ntest: skew': '#C85200', 
			'train: unskew\ntest: unskew': 'black', 
			'train: skew\ntest: unskew': '#ABABAB', 
			'train: unskew\ntest: skew': '#5F6B29'
		}

		res['color'] = res['train_test'].map(color_mapping)

		xlabels = []
		for modelid, model in enumerate(available_models):
			print(f'plot {model}')
			xlabels.append(model_to_spec[model]['label'])
			plot_errorbars_model_groups(plt, res, model, 'auc', barwidth, modelid)

		savename = os.path.join(results_dir,
			f'chexpert_{FLAGS.pval}_{FLAGS.metric}_{FLAGS.groupby}.pdf')
		cropped_savename = os.path.join(results_dir,
			f'chexpert_{FLAGS.pval}_{FLAGS.metric}_{FLAGS.groupby}_cropped.pdf')

		plt.ylim(bottom = MIN_METRIC)
		plt.ylabel('AUROC')

		midpoint = (barwidth * len(available_models))/2.0
		print(midpoint)
		print([i + midpoint for i in range(4)])
		plt.xticks(ticks = [i + midpoint for i in range(4)], labels = xlabels)

		markers = [
			plt.Line2D([0,0],[0,0], color=color, marker='s', linestyle='') for color in color_mapping.values()
			]
		plt.legend(markers, color_mapping.keys(), prop={'size': 12}, ncol = 2, handletextpad=0.05)

		plt.tight_layout()
		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])



if __name__ == '__main__':
	app.run(main)
