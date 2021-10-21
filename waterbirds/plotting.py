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
flags.DEFINE_string('experiment_name', '5050', 'Name of the experiment.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_enum('plot_type', 'main', ['main', 'ablation', 'oracle', 'minimal', 'minimal_dnn'], 'which plot?')
flags.DEFINE_string('clean_back', 'True', 'clean or noisy backgrounds.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

NUM_WORKERS = 30


def get_model_dict(pval, experiment_name, plot_type):
	if (experiment_name == "8050") and (plot_type == 'main'):
		model_specs = {
			f'unweighted_slabs_ts{pval}': {
				'label': r'\textbf{MMD-T}', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': r'\textbf{MMD-S}', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			'simple_baseline_classic': {
				'label': r'\textbf{L2-S}', 'linestyle': 'dotted',
				'color': 'black'
			},
			'rex_classic': {
				'label': r'\textbf{Rex}', 'linestyle': 'dotted',
				'color': '#f0e442'
			},
			f'slabs_unweighted_two_way_ts{pval}': {
				'label': r'\textbf{cMMD-T}', 'linestyle': 'dotted',
				'color': '#cc79a7'
			},
			'random_aug_classic': {
				'label': r'\textbf{Rand-Aug-S}', 'linestyle': 'solid',
				'color': '#5F6B29'
			},

		}


	if (experiment_name == "8090") and (plot_type == 'main'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': r'\textbf{wMMD-T}', 'linestyle': 'solid',
				'color': '#C85200'
			},
			'simple_baseline_classic': {
				'label': r'\textbf{L2-S}', 'linestyle': 'dotted',
				'color': 'black'
			},
			'weighted_baseline_classic': {
				'label': r'\textbf{wL2-S}', 'linestyle': 'solid',
				'color': '#ABABAB'
			},
			'rex_classic': {
				'label': r'\textbf{Rex}', 'linestyle': 'dotted',
				'color': '#f0e442'
			},
			f'slabs_unweighted_two_way_ts{pval}': {
				'label': r'\textbf{cMMD-T}', 'linestyle': 'dotted',
				'color': '#cc79a7'
			},
			'random_aug_classic': {
				'label': r'\textbf{Rand-Aug-S}', 'linestyle': 'solid',
				'color': '#5F6B29'
			},

		}

	if (experiment_name == "8050") and (plot_type == 'minimal'):
		del plot_type
		model_specs = {
			f'unweighted_slabs_ts{pval}': {
				'label': 'Ours', 'linestyle': 'solid',
				'color': '#C85200'
			},
			'simple_baseline_classic': {
				'label': 'L2', 'linestyle': 'dotted',
				'color': 'black'
			},
			'random_aug_classic': {
				'label': 'Rand-Aug', 'linestyle': 'solid',
				'color': '#5F6B29'
			},
		}

	if (experiment_name == "8090") and (plot_type == 'minimal'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': 'Ours', 'linestyle': 'solid',
				'color': '#C85200'
			},
			'simple_baseline_classic': {
				'label': 'L2', 'linestyle': 'dotted',
				'color': 'black'
			},
			'weighted_baseline_classic': {
				'label': 'w-L2', 'linestyle': 'solid',
				'color': '#ABABAB'
			},
			'random_aug_classic': {
				'label': 'Augment', 'linestyle': 'solid',
				'color': '#5F6B29'
			},
# 			f'unweighted_slabs_uts{pval}': {
# 				'label': 'MMD-only', 'linestyle': 'solid',
# 				'color': '#800080'
# 			},

		}


	if (experiment_name == "8090") and (plot_type == 'ablation'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': r'\textbf{wMMD-T}', 'linestyle': 'solid',
				'color': '#C85200'
			},

			f'slabs_weighted_bal_classic': {
				'label': r'\textbf{wMMD-S}', 'linestyle': 'solid',
				'color': '#FFBC79'
			},

			f'unweighted_slabs_ts{pval}': {
				'label': r'\textbf{MMD-T}', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': r'\textbf{MMD-S}', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			f'unweighted_slabs_uts{pval}': {
				'label': r'\textbf{MMD-uT}', 'linestyle': 'solid',
				'color': '#A2C8EC'
			},

		}
	if (experiment_name == "8090") and (plot_type == 'oracle'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': 'wMMD-reg-T', 'linestyle': 'solid',
				'color': '#C85200'
			},

			'oracle_aug_0.1_classic': {
				'label': 'Or-aug-10\%-S', 'linestyle': 'solid',
				'color': '#945634'
			},
			'oracle_aug_0.5_classic': {
				'label': 'Or-aug-50\%-S', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'oracle_aug_1.0_classic': {
				'label': 'Or-aug-100\%-S', 'linestyle': 'solid',
				'color': '#a48b00'
			},
		}

	return model_specs


def plot_errorbars(model_to_spec, axis, legend_elements, results, model, metric):
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
		model_results[f'{metric}_mean'],
		yerr=model_results[f'{metric}_std'] / np.sqrt(20),
		color=model_to_spec[model]['color'],
		# linestyle=model_to_spec[model]['linestyle'],
		linewidth=3,
		label=model_to_spec[model]['label'],
		capsize=5)

	# model_legend_entry = Patch(facecolor=model_to_spec[model]['color'],
	# 	label=model_to_spec[model]['label'])
	# if legend_elements is not None:
	# 	legend_elements.append(model_legend_entry)


def plot_errorclouds(model_to_spec, axis, legend_elements, results, model, metric):
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
	plt.plot(model_results.py1_y0_s,
		model_results[f'{metric}_mean'], '.',
		color=model_to_spec[model]['color'])

	axis.plot(
		model_results.py1_y0_s,
		model_results[f'{metric}_mean'],
		color=model_to_spec[model]['color'],
		label=model_to_spec[model]['label'],
		linewidth=1)

	lower = model_results[f'{metric}_mean'] - model_results[f'{metric}_std']
	upper = model_results[f'{metric}_mean'] + model_results[f'{metric}_std']

	axis.fill_between(model_results.py1_y0_s, lower, upper, alpha=0.1,
			color=model_to_spec[model]['color'], linewidth=2)

	model_legend_entry = Patch(facecolor=model_to_spec[model]['color'],
		label=model_to_spec[model]['label'])
	if legend_elements is not None:
		legend_elements.append(model_legend_entry)


def main(argv):
	del argv
	model_to_spec = get_model_dict(FLAGS.pval, FLAGS.experiment_name,
		FLAGS.plot_type)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	res = []
	for model in model_to_spec:
		try:
			model_res = pd.read_csv((f'{BASE_DIR}/final_models/{model}_'
				f'{FLAGS.experiment_name}_{FLAGS.clean_back}_{FLAGS.batch_size}.csv'))
		except FileNotFoundError as e:
			print(e)
			continue
		res.append(model_res)
	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	available_models = res.model.unique().tolist()

	py1_y0 = float(FLAGS.experiment_name[2:]) / 100.0
	plt.figure(figsize=(7, 5))
	font = {'size': 14, 'family': 'serif', 'serif': 'Computer Modern Roman'}
	plt.rc('font', **font)
	plt.rc('text', usetex=True)

	legend_elements = []
	for model in available_models:
		print(f'plot {model}')
		plot_errorbars(model_to_spec, plt, legend_elements, res, model, 'auc')

	savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'{FLAGS.batch_size}_{FLAGS.plot_type}.pdf'))
	cropped_savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'{FLAGS.batch_size}_{FLAGS.plot_type}_cropped.pdf'))

	plt.axvline(py1_y0, linestyle='--', color='black')
	# if (FLAGS.experiment_name == '8090') and ((FLAGS.plot_type == 'minimal') or (FLAGS.plot_type == 'minimal_dnn')):
	# 	vline_annotate = (r'P(Water bird $|$ water background)'
	# 		'\n'
	# 		r'= P(Land bird $|$ land background) at train time')

	# 	plt.text(0.9, 0.77, vline_annotate,
	# 		horizontalalignment='center',
	# 		verticalalignment='center',
	# 		rotation='vertical',
	# 		fontsize=9)
	plt.xlabel(r'P(Water bird $|$ water background) = P(Land bird $|$ land background)'
						'\n'
						r'at test time')
	plt.ylabel('AUROC')
	if (FLAGS.experiment_name == '8090') and (FLAGS.plot_type == 'ablation'):
		plt.legend(bbox_to_anchor=(0.0, 0.55), loc='lower left', prop={'size': 12})
	elif FLAGS.experiment_name == '8090':
		plt.legend(bbox_to_anchor=(0.55, 0.01), loc='lower left', prop={'size': 12})
	else:
		plt.legend(bbox_to_anchor=(0.7, 0.2), loc='lower left', prop={'size': 12})
	plt.tight_layout()
	plt.savefig(savename)
	plt.clf()
	plt.close()
	crop(["-p", "5", savename, "-o", cropped_savename])

if __name__ == '__main__':
	app.run(main)
