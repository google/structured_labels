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
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_enum('plot_type', 'main', ['main', 'ablation', 'oracle'], 'which plot?')
flags.DEFINE_string('clean_back', 'True', 'clean or noisy backgrounds.')



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

NUM_WORKERS = 30
PLOT_ALL = False
MAIN_PLOT = True

def get_model_dict(pval, experiment_name, plot_type):
	if experiment_name == "8050":
		del plot_type
		model_specs = {
			f'unweighted_slabs_ts{pval}': {
				'label': 'MMD-reg-T', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': 'MMD-reg-C', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			'simple_baseline_classic': {
				'label': 'L2-reg-C', 'linestyle': 'dotted',
				'color': 'black'
			},
			'random_aug_classic': {
				'label': 'Rand-Aug-C', 'linestyle': 'solid',
				'color': '#5F6B29'
			},

		}


	if (experiment_name == "8090") and (plot_type == 'main'):
		model_specs = {
			# f'slabs_weighted_bal_ts{pval}': {
			# 	'label': 'Ours', 'linestyle': 'solid',
			# 	'color': '#C85200'
			# },
			'simple_baseline_classic': {
				'label': 'DNN', 'linestyle': 'dotted',
				'color': 'black'
			},
			# 'weighted_baseline_classic': {
			# 	'label': 'wL2-reg-C', 'linestyle': 'solid',
			# 	'color': '#ABABAB'
			# },

			# 'random_aug_classic': {
			# 	'label': 'Rand-Aug-C', 'linestyle': 'solid',
			# 	'color': '#5F6B29'
			# },

		}

	if (experiment_name == "8090") and (plot_type == 'ablation'):
		model_specs = {
			f'slabs_weighted_bal_ts{pval}': {
				'label': 'wMMD-reg-T', 'linestyle': 'solid',
				'color': '#C85200'
			},

			f'slabs_weighted_bal_classic': {
				'label': 'wMMD-reg-C', 'linestyle': 'solid',
				'color': '#FFBC79'
			},

			f'unweighted_slabs_ts{pval}': {
				'label': 'MMD-reg-T', 'linestyle': 'solid',
				'color': '#006BA4'
			},
			'unweighted_slabs_classic': {
				'label': 'MMD-reg-C', 'linestyle': 'solid',
				'color': '#5F9ED1'
			},

			f'unweighted_slabs_uts{pval}': {
				'label': 'MMD-reg-uT', 'linestyle': 'solid',
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
	model_to_spec = get_model_dict(FLAGS.pval, FLAGS.experiment_name, FLAGS.plot_type)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	if PLOT_ALL:
		res = []
		for model in model_to_spec.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{FLAGS.experiment_name}_{FLAGS.clean_back}.csv')
			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)
		res = pd.concat(res, axis=0, ignore_index=True, sort=False)
		available_models = res.model.unique().tolist()

		_, axes = plt.subplots(1, 2, figsize=(10, 5))
		legend_elements = []

		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(model_to_spec, axes[0], legend_elements, res, model, 'auc')
			plot_errorbars(model_to_spec, axes[1], None, res, model, 'accuracy')
			# plot_errorbars(model_to_spec, axes[2], None, res, model, 'loss')

		py0 = float(FLAGS.experiment_name[:2]) / 100.0
		py1_y0 = float(FLAGS.experiment_name[2:]) / 100.0

		axes[0].set_xlabel('Conditional probability in shifted distribution')
		axes[0].set_ylabel('AUC')
		axes[0].axvline(py1_y0, linestyle='--', color='black')
		# axes[0].legend(handles=legend_elements, loc='lower right')
		axes[1].set_xlabel('Conditional probability in shifted distribution')
		axes[1].set_ylabel('Accuracy')
		axes[1].axvline(py1_y0, linestyle='--', color='black')
		# axes[1].legend(handles=legend_elements, loc='upper left')
		axes[1].legend(loc='upper left')
		# axes[2].set_xlabel('Conditional probability in shifted distribution')
		# axes[2].set_ylabel('Binary cross-entropy (Loss)')
		# axes[2].axvline(0.95, linestyle='--', color='black')
		# axes[2].legend(handles=legend_elements, loc='upper right')
		plt.suptitle(f'P(water bird) = {py0}, P(bird | background) = {py1_y0}')
		plt.savefig(os.path.join(results_dir,
			f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_plot.pdf'))
		plt.clf()
		plt.close()

	elif MAIN_PLOT:
		res = []
		for model in model_to_spec.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{FLAGS.experiment_name}_{FLAGS.clean_back}.csv')
			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)
		res = pd.concat(res, axis=0, ignore_index=True, sort=False)
		available_models = res.model.unique().tolist()

		py1_y0 = float(FLAGS.experiment_name[2:]) / 100.0
		plt.figure(figsize=(7, 5))
		font = {'size': 16, 'family': 'serif', 'serif': 'Computer Modern Roman'}
		plt.rc('font', **font)
		plt.rc('text', usetex=True)

		legend_elements = []
		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(model_to_spec, plt, legend_elements, res, model, 'auc')

		# savename = os.path.join(results_dir,
		# 	f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_{FLAGS.plot_type}.pdf')
		# cropped_savename = os.path.join(results_dir,
		# 	f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_{FLAGS.plot_type}_cropped.pdf')

		savename = os.path.join(results_dir, f'for_alex_dnn.pdf')
		print(savename)
		cropped_savename = os.path.join(results_dir, f'cropped_for_alex_dnn.pdf')
		plt.axvline(py1_y0, linestyle='--', color='black')
		plt.xlabel(r'P(Water bird $|$ water background) = P(land bird $|$ land background) at test time')
		plt.ylabel('AUROC')
		if (FLAGS.experiment_name == '8090') and (FLAGS.plot_type == 'ablation'):
			plt.legend(bbox_to_anchor=(0.0, 0.55), loc='lower left', prop={'size': 12})
		elif FLAGS.experiment_name == '8090':
			plt.legend(bbox_to_anchor=(0.55, 0.01), loc='lower left', prop={'size': 12})
		else:
			plt.legend(bbox_to_anchor=(0.7, 0.0), loc='lower left', prop={'size': 12})
		plt.tight_layout()
		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])

	else:
		# -- import baseline and ours
		baseline_res = pd.read_csv(
			f'{BASE_DIR}/final_models/simple_baseline_classic_per_run.csv'
		)

		baseline_mean = pd.read_csv(
			f'{BASE_DIR}/final_models/simple_baseline_classic.csv'
		)
		baseline_mean['random_seed'] = -1
		baseline_mean = baseline_mean[baseline_res.columns.tolist()]
		baseline_res = pd.concat([baseline_res, baseline_mean], axis=0)

		ours_res = pd.read_csv(
			f'{BASE_DIR}/final_models/slabs_non_equivalent_per_run.csv'
		)

		ours_mean = pd.read_csv(
			f'{BASE_DIR}/final_models/slabs_non_equivalent.csv'
		)

		ours_mean['random_seed'] = -1
		ours_mean = ours_mean[ours_res.columns.tolist()]
		ours_res = pd.concat([ours_res, ours_mean], axis=0)

		# -- start plot
		font = {'size': 22, 'family': 'serif', 'serif': 'Computer Modern Roman'}
		plt.rc('font', **font)
		plt.rc('text', usetex=True)
		_, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

		# -- plot baseline
		# baseline_worst = baseline_res[(baseline_res.py1_y0_s == baseline_res.py1_y0_s.max())]
		# baseline_worst = baseline_worst[
		# 	(baseline_worst.accuracy_mean == baseline_worst.accuracy_mean.min())
		# ]

		# baseline_random_seed = baseline_worst.random_seed.values[0]
		# print(baseline_random_seed)
		for random_seed in baseline_res.random_seed.unique():
			res_rs = baseline_res[(baseline_res.random_seed == random_seed)]
			alpha = 1 if random_seed == -1 else 0.2
			axes[0].plot(model_to_spec, res_rs.py1_y0_s, res_rs.accuracy_mean, linewidth=1, alpha=alpha, color='black')
			axes[0].plot(model_to_spec, res_rs.py1_y0_s, res_rs.accuracy_mean, 'o', alpha=alpha, color='black')
		axes[0].set_xlabel(r'P(Water bird $|$ water background) at test time')
		axes[0].set_ylabel('Accuracy')
		axes[0].title.set_text(r'DNN')

		# -- plot ours
		# ours_worst = ours_res[(ours_res.py1_y0_s == ours_res.py1_y0_s.max())]
		# ours_worst = ours_worst[
		# 	(ours_worst.accuracy_mean == ours_worst.accuracy_mean.min())
		# ]
		# ours_random_seed = ours_worst.random_seed.values[0]

		for random_seed in ours_res.random_seed.unique():
			res_rs = ours_res[(ours_res.random_seed == random_seed)]
			alpha = 1 if random_seed == -1 else 0.2
			axes[1].plot(model_to_spec, res_rs.py1_y0_s, res_rs.accuracy_mean, linewidth=1, alpha=alpha, color='black')
			axes[1].plot(model_to_spec, res_rs.py1_y0_s, res_rs.accuracy_mean, 'o', alpha=alpha, color='black')
		axes[1].set_xlabel(r'P(Water bird $|$ water background) at test time')
		axes[1].title.set_text(r'iSlabs')


		plt.tight_layout()

		savename = os.path.join(results_dir, f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_plot_per_run.pdf')

		cropped_savename = os.path.join(results_dir,
			f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_plot_per_run_cropped.pdf')

		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])


if __name__ == '__main__':
	app.run(main)
