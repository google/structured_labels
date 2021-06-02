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
import matplotlib 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfCropMargins import crop
plt.style.use('tableau-colorblind10')

FLAGS = flags.FLAGS
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_string('clean_back', 'True', 'clean or noisy backgrounds.')


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))



def get_model_dict(pval, experiment_name, plot_type):
	if experiment_name == "8050":
		del plot_type
		model_specs = {
			f'unweighted_slabs_ts{pval}': {
				'label': r'\textbf{MMD-reg-T}', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': r'\textbf{MMD-reg-S}', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			'simple_baseline_classic': {
				'label': r'\textbf{L2-reg-S}', 'linestyle': 'dotted',
				'color': 'black'
			},
			'random_aug_classic': {
				'label': r'\textbf{Rand-Aug-S}', 'linestyle': 'solid',
				'color': '#5F6B29'
			},

		}


	if (experiment_name == "8090") and (plot_type == 'main'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': r'\textbf{wMMD-reg-T}', 'linestyle': 'solid',
				'color': '#C85200'
			},
			'simple_baseline_classic': {
				'label': r'\textbf{L2-reg-S}', 'linestyle': 'dotted',
				'color': 'black'
			},
			'weighted_baseline_classic': {
				'label': r'\textbf{wL2-reg-S}', 'linestyle': 'solid',
				'color': '#ABABAB'
			},

			'random_aug_classic': {
				'label': r'\textbf{Rand-Aug-S}', 'linestyle': 'solid',
				'color': '#5F6B29'
			},

		}

	if (experiment_name == "8090") and (plot_type == 'ablation'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': r'\textbf{wMMD-reg-T}', 'linestyle': 'solid',
				'color': '#C85200'
			},

			f'slabs_weighted_bal_classic': {
				'label': r'\textbf{wMMD-reg-S}', 'linestyle': 'solid',
				'color': '#FFBC79'
			},

			f'unweighted_slabs_ts{pval}': {
				'label': r'\textbf{MMD-reg-T}', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': r'\textbf{MMD-reg-S}', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			f'unweighted_slabs_uts{pval}': {
				'label': r'\textbf{MMD-reg-uT}', 'linestyle': 'solid',
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
				'label': 'Or-aug-10\%-C', 'linestyle': 'solid',
				'color': '#945634'
			},
			'oracle_aug_0.5_classic': {
				'label': 'Or-aug-50\%-C', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'oracle_aug_1.0_classic': {
				'label': 'Or-aug-100\%-C', 'linestyle': 'solid',
				'color': '#a48b00'
			},
		}

	return model_specs


def plot_errorbars(model_to_spec, axis, results, model, metric):
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
	axis.errorbar(
		model_results.py1_y0_s,
		model_results[f'{metric}_mean'],
		yerr=model_results[f'{metric}_std'] / np.sqrt(20),
		color=model_to_spec[model]['color'],
		# linestyle=model_to_spec[model]['linestyle'],
		linewidth=3,
		# label=model_to_spec[model]['label'],
		capsize=5)


def main(argv):
	del argv
	

	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	matplotlib.rcParams['text.usetex'] = True
	matplotlib.rcParams['font.family'] = 'serif'
	matplotlib.rcParams['font.weight'] = 'extra bold'
	matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
	matplotlib.rcParams['axes.labelsize'] = 32
	matplotlib.rcParams['ytick.labelsize'] = 24
	matplotlib.rcParams['xtick.labelsize'] = 24
	matplotlib.rcParams["legend.columnspacing"] = 1

	legend_text_size = 20

	fig, axes = plt.subplots(1, 3, figsize=(21, 6))
	# font = {'size': 22, 'family': 'serif', 'serif': 'Computer Modern Roman'}


	# --- loop to ploit the first two subplots
	for axid, experiment_name in enumerate(["8050", "8090"]):
		model_to_spec = get_model_dict(FLAGS.pval, experiment_name, 'main')
		
		# -- collect results 
		res = []
		for model in model_to_spec.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{experiment_name}_{FLAGS.clean_back}.csv')
			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)
		res = pd.concat(res, axis=0, ignore_index=True, sort=False)
		available_models = res.model.unique().tolist()
		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(model_to_spec, axes[axid], res, model, 'auc')

		py1_y0 = float(experiment_name[2:]) / 100.0
		axes[axid].axvline(py1_y0, linestyle='--', color='black')

		markers = [
			plt.Line2D([0,0],[0,0], color=model_to_spec[model]['color'], 
				marker='s', linestyle='', markersize=12) for model in available_models
			]
		

		# ylabels = axes[axid].get_yticks()
		# ylabels = [f'{lab:.02f}' for lab in ylabels]
		# ylabels = [r'\textbf{' + lab + '}' for lab in ylabels]
		# axes[axid].set_yticklabels(labels = ylabels)


		axes[axid].set_xticks(ticks = [0.2, 0.4, 0.6, 0.8])
		axes[axid].set_xticklabels(labels = [r'\textbf{0.2}', 
			r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}'])


		if experiment_name == '8090':
			# axes[axid].legend(markers, [model_to_spec[model]['label'] for model in available_models],
			# 	bbox_to_anchor=(0.45, 0.01), loc='lower left', prop={'size': legend_text_size},  handletextpad=0.05)
			axes[axid].set_yticks(ticks = [0.7, 0.8, 0.9])
			axes[axid].set_yticklabels(labels = [r'\textbf{0.7}', r'\textbf{0.8}', r'\textbf{0.9}'])

			axes[axid].legend(markers, [model_to_spec[model]['label'] for model in available_models],
				bbox_to_anchor=(0.5, 1.0), loc='lower center', prop={'size': legend_text_size},  handletextpad=0.05, ncol=2)
			axes[axid].set_xlabel(
				r'\textbf{P(Water bird $|$ water background) = P(Land bird $|$ land background) at test time}')
		else:
			axes[axid].set_yticks(ticks = [0.80, 0.85, 0.90])
			axes[axid].set_yticklabels(labels = [r'\textbf{0.80}', r'\textbf{0.85}', r'\textbf{0.90}'])

			# axes[axid].legend(markers, [model_to_spec[model]['label'] for model in available_models], 
			# 	bbox_to_anchor=(0.5, -0.05), loc='lower center', prop={'size': legend_text_size}, 
			# 	ncol=2,  handletextpad=0.05)
			axes[axid].legend(markers, [model_to_spec[model]['label'] for model in available_models], 
				bbox_to_anchor=(0.5, 1.0), loc='lower center', prop={'size': legend_text_size}, 
				ncol=2,  handletextpad=0.05)
			axes[axid].set_ylabel(r'\textbf{AUROC}')


	# --- plot the ablation study
	experiment_name = "8090"

	model_to_spec = get_model_dict(FLAGS.pval, experiment_name, 'ablation')	
	# -- collect results 
	res = []
	for model in model_to_spec.keys():
		try:
			model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{experiment_name}_{FLAGS.clean_back}.csv')
		except FileNotFoundError as e:
			print(e)
			continue
		res.append(model_res)
	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	available_models = res.model.unique().tolist()

	
	for model in available_models:
		print(f'plot {model}')
		plot_errorbars(model_to_spec, axes[-1], res, model, 'auc')

	markers = [
		plt.Line2D([0,0],[0,0], color=model_to_spec[model]['color'], 
			marker='s', linestyle='', markersize=12) for model in available_models
		]

	py1_y0 = float(experiment_name[2:]) / 100.0
	axes[-1].axvline(py1_y0, linestyle='--', color='black')
	# axes[-1].legend(markers, [model_to_spec[model]['label'] for model in available_models],
	# 	bbox_to_anchor=(0.0, 1), loc='upper left', prop={'size': legend_text_size},
	# 	ncol=2,  handletextpad=0.05)
	axes[-1].legend(markers, [model_to_spec[model]['label'] for model in available_models], 
		bbox_to_anchor=(0.5, 1.0), loc='lower center', prop={'size': legend_text_size}, 
		ncol=2,  handletextpad=0.05)


	axes[-1].set_xticks(ticks = [0.2, 0.4, 0.6, 0.8])
	axes[-1].set_xticklabels(labels = [r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}'])

	ylabels = axes[-1].get_yticks()
	ylabels = [f'{lab:.02f}' for lab in ylabels]
	ylabels = [r'\textbf{' + lab + '}' for lab in ylabels]
	axes[-1].set_yticklabels(labels = ylabels)




	savename = os.path.join(results_dir,
		f'waterbirds_combined_plot_{FLAGS.pval}_{FLAGS.clean_back}.pdf')
	cropped_savename = os.path.join(results_dir,
		f'waterbirds_combined_plot_{FLAGS.pval}_{FLAGS.clean_back}_cropped.pdf')

	plt.tight_layout()
	plt.savefig(savename)
	plt.clf()
	plt.close()
	crop(["-p4", "5", "5", "5", "55", savename, "-o", cropped_savename])

if __name__ == '__main__':
	app.run(main)
