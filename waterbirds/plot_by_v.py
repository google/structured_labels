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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfCropMargins import crop
import time, datetime
plt.style.use('tableau-colorblind10')

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_name', '8090', 'Name of the experiment.')
flags.DEFINE_string('metric', 'auc', 'metric: auc or acc.')
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_string('clean_back', 'True', 'clean or noisy backgrounds.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

BAR_WIDTH = 0.35


def get_model_dict(pval):
	model_specs = {
		f'slabs_weighted_bal_ts{pval}': {
			'label': 'PM', 'linestyle': 'solid',
			'color': '#C85200'
		},
		f'slabs_unweighted_two_way_ts{pval}': {
			'label': 'EO', 'linestyle': 'solid',
			'color': '#006BA4'
		},
		f'unweighted_slabs_uts{pval}': {
			'label': 'DP', 'linestyle': 'solid',
			'color': '#A2C8EC'
		},
		'simple_baseline_classic': {
			'label': 'DNN', 'linestyle': 'dotted',
			'color': '#5F6B29'
		},
		# 'rex_classic': {
		# 	'label': 'ReX', 'linestyle': 'dotted',
		# 	'color': '#800080'
		# },
	}
	return model_specs

def plot_bars(model_results, axis, location, metric):
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
	if metric == 'auc':
		axis.bar(location - BAR_WIDTH / 2, model_results['acc_v0']['mean'],
			yerr=model_results['auroc_v0']['std'] / np.sqrt(10),
			width=BAR_WIDTH, color='#A2C8EC', capsize=5)
		axis.bar(location + BAR_WIDTH / 2, model_results['acc_v1']['mean'],
			yerr=model_results['auroc_v1']['std'] / np.sqrt(10),
			width=BAR_WIDTH, color='#C85200', capsize=5)
	else:
		axis.bar(location - 3 * BAR_WIDTH / 4, model_results['acc_00']['mean'],
			yerr=model_results['acc_00']['std'], capsize=5,
			width=BAR_WIDTH / 2, color='#C85200')
		axis.bar(location - BAR_WIDTH / 4, model_results['acc_01']['mean'],
			yerr=model_results['acc_01']['std'], capsize=5,
			width=BAR_WIDTH / 2, color='#A2C8EC')

		axis.bar(location + 3 * BAR_WIDTH / 4, model_results['acc_10']['mean'],
			yerr=model_results['acc_10']['std'], capsize=5,
			width=BAR_WIDTH / 2, color='#5F6B29')
		axis.bar(location + BAR_WIDTH / 4, model_results['acc_11']['mean'],
			yerr=model_results['acc_11']['std'], capsize=5,
			width=BAR_WIDTH / 2, color='#006BA4')


def main(argv):
	del argv
	model_to_spec = get_model_dict(FLAGS.pval)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	# --- start figure
	plt.figure(figsize=(9, 5))
	font = {'size': 14, 'family': 'serif', 'serif': 'Computer Modern Roman'}
	plt.rc('font', **font)
	plt.rc('text', usetex=True)

	model_names = []
	for model_id, model in enumerate(model_to_spec):
		# --- get the model performance
		try:
			model_file = (f'{BASE_DIR}/final_models/v_performance_{model}_'
				f'{FLAGS.experiment_name}_{FLAGS.clean_back}_{FLAGS.batch_size}.csv')
			model_res = pd.read_csv(model_file,index_col=0)
			file_date = time.ctime(os.path.getmtime(model_file))
			file_date = datetime.datetime.strptime(file_date, "%a %b %d %H:%M:%S %Y")
			file_date = file_date.strftime('%Y-%m-%d %H:%M:%S')
			print(f"{model} last modifid: {file_date}")

		except FileNotFoundError as e:
			print(e)
			continue
		plot_bars(model_res, plt, model_id, FLAGS.metric)
		model_names.append(model_to_spec[model]['label'])

	if FLAGS.metric == 'auc':
		legend_elements = [  # noqa: F841
			Line2D([0], [0], marker='s', color='#A2C8EC', label='Water background',
				lw=0, markerfacecolor='#A2C8EC'),
			Line2D([0], [0], marker='s', color='#C85200', label='Land background',
				lw=0, markerfacecolor='#C85200')
		]
	else:
		legend_elements = [  # noqa: F841
			Line2D([0], [0], marker='s', color='#C85200',
				label='Land bird\nLand background',
				lw=0, markerfacecolor='#C85200'),
			Line2D([0], [0], marker='s', color='#A2C8EC',
				label='Water bird\nLand background',
				lw=0, markerfacecolor='#A2C8EC'),
			Line2D([0], [0], marker='s', color='#5F6B29',
				label='Land bird\nWater background',
				lw=0, markerfacecolor='#5F6B29'),
			Line2D([0], [0], marker='s', color='#006BA4',
				label='Water bird\nWater background',
				lw=0, markerfacecolor='#006BA4')
		]

	if FLAGS.metric == 'auc':
		if 'rex_classic' in model_to_spec.keys():
			plt.ylim(bottom=0.65)
		else: 
			plt.ylim(bottom=0.7)
		plt.ylabel('Accuracy')

	else:
		plt.ylabel('Accuracy')

	plt.xticks(range(len(model_to_spec)), model_names)
	plt.xlabel('Model')

	plt.legend(handles=legend_elements,
		loc='center left', bbox_to_anchor=(1.04, 0.5),
		handletextpad=0.01, borderaxespad=0,
		fontsize=12)
	plt.tight_layout(rect=[0, 0, 1.01, 1])

	savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'v_{FLAGS.metric}.pdf'))
	cropped_savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'v_{FLAGS.metric}_cropped.pdf'))

	plt.savefig(savename)
	plt.clf()
	plt.close()
	crop(["-p", "5", savename, "-o", cropped_savename])

if __name__ == '__main__':
	app.run(main)


