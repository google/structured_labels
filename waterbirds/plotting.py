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


FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', '5050', 'Name of the experiment.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

NUM_WORKERS = 30
PLOT_ALL = True
MAIN_PLOT = False
NUM_REPS = 10

MODEL_TO_PLOT_SPECS = {
	'slabs_classic': {
		'color': '#377eb8', 'label': 'Ours', 'linestyle': 'solid'
	},
	# 'slabs_ts': {
	# 	'color': '#a65628', 'label': 'Ours (ts)', 'linestyle': 'solid'
	# },

	# 'unweighted_slabs_logit_classic': {
	# 	'color': '#b8373e', 'label': 'Unweighted ours (logit)', 'linestyle': 'solid'
	# },
	'unweighted_slabs_classic': {
		'color': '#b8373e', 'label': 'Unweighted ours', 'linestyle': 'solid'
	},

	'weighted_baseline_classic': {
		'color': '#4daf4a', 'label': 'W-DNN', 'linestyle': 'solid'
	},

	'simple_baseline_classic': {
		'color': '#f781bf', 'label': 'DNN', 'linestyle': 'solid'
	},
}


def plot_errorbars(axis, legend_elements, results, model, metric):
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
		yerr=model_results[f'{metric}_std']/np.sqrt(10),
		color=MODEL_TO_PLOT_SPECS[model]['color'],
		linestyle=MODEL_TO_PLOT_SPECS[model]['linestyle'],
		capsize=5)

	model_legend_entry = Patch(facecolor=MODEL_TO_PLOT_SPECS[model]['color'],
		label=MODEL_TO_PLOT_SPECS[model]['label'])
	if legend_elements is not None:
		legend_elements.append(model_legend_entry)


def plot_errorclouds(axis, legend_elements, results, model, metric):
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
		color=MODEL_TO_PLOT_SPECS[model]['color'])

	axis.plot(
		model_results.py1_y0_s,
		model_results[f'{metric}_mean'],
		color=MODEL_TO_PLOT_SPECS[model]['color'],
		linestyle=MODEL_TO_PLOT_SPECS[model]['linestyle'],
		linewidth=1)

	lower = model_results[f'{metric}_mean'] - model_results[f'{metric}_std']/np.sqrt(10)
	upper = model_results[f'{metric}_mean'] + model_results[f'{metric}_std']/np.sqrt(10)

	axis.fill_between(model_results.py1_y0_s, lower, upper, alpha=0.1,
			color=MODEL_TO_PLOT_SPECS[model]['color'], linewidth=2)

	model_legend_entry = Patch(facecolor=MODEL_TO_PLOT_SPECS[model]['color'],
		label=MODEL_TO_PLOT_SPECS[model]['label'])
	if legend_elements is not None:
		legend_elements.append(model_legend_entry)


def main(argv):
	del argv
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	if PLOT_ALL:
		res = []
		for model in MODEL_TO_PLOT_SPECS.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_{FLAGS.exp_name}.csv')
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
			plot_errorbars(axes[0], legend_elements, res, model, 'auc')
			plot_errorbars(axes[1], None, res, model, 'accuracy')
			# plot_errorbars(axes[2], None, res, model, 'loss')

		py0 = float(FLAGS.exp_name[:2]) / 100.0
		py1_y0 = float(FLAGS.exp_name[2:]) / 100.0

		axes[0].set_xlabel('Conditional probability in shifted distribution')
		axes[0].set_ylabel('AUC')
		axes[0].axvline(py1_y0, linestyle='--', color='black')
		axes[0].legend(handles=legend_elements, loc='lower right')


		axes[1].set_xlabel('Conditional probability in shifted distribution')
		axes[1].set_ylabel('Accuracy')
		axes[1].axvline(py1_y0, linestyle='--', color='black')
		# axes[1].legend(handles=legend_elements, loc='upper left')

		# axes[2].set_xlabel('Conditional probability in shifted distribution')
		# axes[2].set_ylabel('Binary cross-entropy (Loss)')
		# axes[2].axvline(0.95, linestyle='--', color='black')
		# axes[2].legend(handles=legend_elements, loc='upper right')


		plt.suptitle(f'P(water bird) = {py0}, P(bird | background) = {py1_y0}')
		plt.savefig(os.path.join(results_dir, f'waterbirds_{FLAGS.exp_name}_plot.pdf'))
		plt.clf()
		plt.close()

	elif MAIN_PLOT:
		res = []
		for model in MODEL_TO_PLOT_SPECS.keys():
			try:
				model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}.csv')
			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)
		res = pd.concat(res, axis=0, ignore_index=True, sort=False)
		available_models = res.model.unique().tolist()

		plt.figure(figsize=(8, 5))
		font = {'size': 22, 'family': 'serif', 'serif': 'Computer Modern Roman'}
		plt.rc('font', **font)
		plt.rc('text', usetex=True)

		legend_elements = []
		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(plt, legend_elements, res, model, 'accuracy')

		savename = os.path.join(results_dir, f'waterbirds_{FLAGS.exp_name}_plot.pdf')
		cropped_savename = os.path.join(results_dir,
			f'waterbirds_{FLAGS.exp_name}_plot_cropped.pdf')

		plt.axvline(0.95, linestyle='--', color='black')
		plt.xlabel(r'P(Water bird $|$ water background)')
		plt.ylabel('Accuracy')
		plt.legend(handles=legend_elements, loc='lower right', prop={'size': 12})
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
			axes[0].plot(res_rs.py1_y0_s, res_rs.accuracy_mean, linewidth=1, alpha=alpha, color='black')
			axes[0].plot(res_rs.py1_y0_s, res_rs.accuracy_mean, 'o', alpha=alpha, color='black')
		axes[0].set_xlabel(r'P(Water bird $|$ water background)')
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
			axes[1].plot(res_rs.py1_y0_s, res_rs.accuracy_mean, linewidth=1, alpha=alpha, color='black')
			axes[1].plot(res_rs.py1_y0_s, res_rs.accuracy_mean, 'o', alpha=alpha, color='black')
		axes[1].set_xlabel(r'P(Water bird $|$ water background)')
		axes[1].title.set_text(r'iSlabs')


		plt.tight_layout()

		savename = os.path.join(results_dir, f'waterbirds_{FLAGS.exp_name}_plot_per_run.pdf')
		cropped_savename = os.path.join(results_dir,
			f'waterbirds_{FLAGS.exp_name}_plot_per_run_cropped.pdf')

		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])


if __name__ == '__main__':
	app.run(main)
