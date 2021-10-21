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
import time, datetime
plt.style.use('tableau-colorblind10')

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_name', '8090', 'Name of the experiment.')
flags.DEFINE_float('pval', 0.05, 'Pvalue.')
flags.DEFINE_enum('metric', 'auc', ['auc', 'accuracy'], 'which metric?')
flags.DEFINE_string('clean_back', 'True', 'clean or noisy backgrounds.')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))

NUM_WORKERS = 30
BATCH_SIZES = [16, 32, 64, 128]
PY1_Y0_S = 0.1

def get_model_dict(pval):
	model_specs = {
		f'slabs_weighted_bal_ts{pval}': {
			'label': 'MMD-T', 'linestyle': 'solid',
			'color': '#C85200'
		},
		# f'unweighted_slabs_uts{pval}': {
		# 	'label': 'DP', 'linestyle': 'solid',
		# 	'color': '#A2C8EC'
		# },
		# f'slabs_weighted_bal_two_way_ts{pval}': {
		# 	'label': 'wEO', 'linestyle': 'solid',
		# 	'color': '#5F6B29'
		# },
		f'slabs_unweighted_two_way_ts{pval}': {
			'label': 'cMMD-T', 'linestyle': 'solid',
			'color': '#006BA4'
		}
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
		model_results.batch_size,
		model_results[f'{metric}_mean'],
		yerr=model_results[f'{metric}_std'] / np.sqrt(20),
		color=model_to_spec[model]['color'],
		linewidth=3,
		label=model_to_spec[model]['label'],
		capsize=5)


def main(argv):
	del argv
	model_to_spec = get_model_dict(FLAGS.pval)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')

	res = []
	for model in model_to_spec:
		for batch_size in BATCH_SIZES:
			try:
				model_file = (f'{BASE_DIR}/final_models/{model}_'
					f'{FLAGS.experiment_name}_{FLAGS.clean_back}_{batch_size}.csv')
				model_res = pd.read_csv(model_file)
				model_res = model_res[(model_res.py1_y0_s == PY1_Y0_S)]
				model_res['batch_size'] = str(batch_size)

				file_date = time.ctime(os.path.getmtime(model_file))
				file_date = datetime.datetime.strptime(file_date, "%a %b %d %H:%M:%S %Y")
				file_date = file_date.strftime('%Y-%m-%d %H:%M:%S')
				print(f"{model} last modifid: {file_date}")

			except FileNotFoundError as e:
				print(e)
				continue
			res.append(model_res)

	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	res.drop('py1_y0_s', axis=1, inplace=True)
	available_models = res.model.unique().tolist()

	# --- start figure
	plt.figure(figsize=(7, 5))
	font = {'size': 14, 'family': 'serif', 'serif': 'Computer Modern Roman'}
	plt.rc('font', **font)
	plt.rc('text', usetex=True)

	legend_elements = []
	for model in available_models:
		print(f'plot {model}')
		plot_errorbars(model_to_spec, plt, legend_elements, res, model, FLAGS.metric)

	savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'{FLAGS.metric}_batch_experiment.pdf'))
	cropped_savename = os.path.join(results_dir,
		(f'waterbirds_{FLAGS.experiment_name}_{FLAGS.pval}_{FLAGS.clean_back}_'
			f'{FLAGS.metric}_batch_experiment_cropped.pdf'))

	plt.xlabel('Batch size')
	if FLAGS.metric == 'accuracy':
		plt.ylabel('Accuracy')
	else: 
		plt.ylabel('AUROC')
	plt.legend()
	plt.tight_layout()
	plt.savefig(savename)
	plt.clf()
	plt.close()
	crop(["-p", "5", savename, "-o", cropped_savename])

if __name__ == '__main__':
	app.run(main)

