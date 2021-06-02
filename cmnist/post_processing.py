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
flags.DEFINE_string('exp_name', 'cmnist', 'Name of the experiment.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))

NUM_WORKERS = 30
PLOT_LOSS = False
NUM_REPS = 10
# https://gist.github.com/thriveth/8560036

MODEL_TO_PLOT_SPECS = {
	# 'slabs_main': {
	# 	'color': '#ff7f0e', 'label': 'iSlabs (ours)', 'linestyle': 'solid'
	# },
	'slabs_classic': {
		'color': '#377eb8', 'label': 'iSlabs-CXV', 'linestyle': 'solid'
	},
	'slabs_non_equivalent': {
		'color': '#a65628', 'label': 'iSlabs (ours)', 'linestyle': 'solid'
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
		yerr=model_results[f'{metric}_std'] / np.sqrt(NUM_REPS),
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

	lower = model_results[f'{metric}_mean'] - model_results[f'{metric}_std'] / np.sqrt(NUM_REPS)
	upper = model_results[f'{metric}_mean'] + model_results[f'{metric}_std'] / np.sqrt(NUM_REPS)

	axis.fill_between(model_results.py1_y0_s, lower, upper, alpha=0.2,
			color=MODEL_TO_PLOT_SPECS[model]['color'], linewidth=1)

	model_legend_entry = Patch(facecolor=MODEL_TO_PLOT_SPECS[model]['color'],
		label=MODEL_TO_PLOT_SPECS[model]['label'])
	if legend_elements is not None:
		legend_elements.append(model_legend_entry)


def main(argv):
	del argv
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

	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	if PLOT_LOSS:
		_, axes = plt.subplots(1, 2, figsize=(14, 5))
		legend_elements = []

		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(axes[0], legend_elements, res, model, 'accuracy')
			plot_errorbars(axes[1], None, res, model, 'loss')

		axes[0].set_xlabel('Conditional probability in shifted distribution')
		axes[0].set_ylabel('Accuracy')
		axes[0].legend(handles=legend_elements, loc='lower right')

		axes[1].set_xlabel('Conditional probability in shifted distribution')
		axes[1].set_ylabel('Loss')
		axes[1].legend(handles=legend_elements, loc='upper right')
		plt.savefig(os.path.join(results_dir, f'{FLAGS.exp_name}_plot.pdf'))
		plt.clf()
		plt.close()

	else:
		plt.figure(figsize=(8, 5))
		font = {'size': 22, 'family': 'serif', 'serif': 'Computer Modern Roman'}
		plt.rc('font', **font)
		plt.rc('text', usetex=True)

		legend_elements = []
		for model in available_models:
			print(f'plot {model}')
			plot_errorbars(plt, legend_elements, res, model, 'accuracy')

		plt.axvline(0.98, linestyle='--', color='black')
		# plt.text(0.91, 0.94, 'Training distribution', rotation=90)
		legend_elements.append(
			Line2D([0], [0], color='black', linestyle='dashed',
				label='Training distribution', lw=2)
		)

		plt.xlabel(r'P(Digit 3 $|$ magenta corruptions)')
		plt.ylabel('Accuracy')
		plt.legend(handles=legend_elements, bbox_to_anchor=(0.8, 0.01),
			loc='lower right', prop={'size': 12})
		plt.tight_layout()

		savename = os.path.join(results_dir, f'{FLAGS.exp_name}_plot_uncropped.pdf')
		cropped_savename = os.path.join(results_dir,
			f'{FLAGS.exp_name}_plot.pdf')

		plt.savefig(savename)
		plt.clf()
		plt.close()
		crop(["-p", "5", savename, "-o", cropped_savename])

if __name__ == '__main__':
	app.run(main)