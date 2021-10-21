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
		f'unweighted_slabs_uts{pval}': {
				'label': 'DP', 'linestyle': 'solid',
				'color': '#A2C8EC'
		},
		f'slabs_unweighted_two_way_ts{pval}': {
			'label': 'EO', 'linestyle': 'solid',
			'color': '#C85200'
		},
		'rex_classic': {
			'label': 'Rex', 'linestyle': 'solid',
			'color': '#006BA4'
		},
	}


	return model_specs


def main(argv):
	del argv
	model_to_spec = get_model_dict(FLAGS.pval)
	results_dir = os.path.join(BASE_DIR, 'results')
	if not os.path.exists(results_dir):
		os.system(f'mkdir -p {results_dir}')


	# --- collect the results 
	res = []

	for model in model_to_spec.keys():
		try:
			model_res = pd.read_csv(f'{BASE_DIR}/final_models/{model}_skew_train.csv')
			# TODO: remove this part 
			model_res = model_res[((model_res.py1_y0_s == 0.5) | (model_res.py1_y0_s == 0.9))]
		except FileNotFoundError as e:
			print(e)
			continue
		res.append(model_res)

	res = pd.concat(res, axis=0, ignore_index=True, sort=False)


	# --- prepare to print ----#	
	# get standard error
	res[f'{FLAGS.metric}_std'] = res[f'{FLAGS.metric}_std']/np.sqrt(5)

	# get the rounded values 
	res[f'{FLAGS.metric} (std)'] = res[f'{FLAGS.metric}_mean'].round(2).astype(str) 
	res[f'{FLAGS.metric} (std)'] = res[f'{FLAGS.metric} (std)'] + " (" 
	res[f'{FLAGS.metric} (std)'] = res[f'{FLAGS.metric} (std)'] + res[f'{FLAGS.metric}_std'].round(3).astype(str) 
	res[f'{FLAGS.metric} (std)'] = res[f'{FLAGS.metric} (std)'] + ")" 

	# reshape long
	res = res[(res.py1_y0_s == 0.5)].merge(res[(res.py1_y0_s == 0.9)],
		on='model', suffixes = ['_shift', '_no_shift'])

	# clean up model names 
	available_models = res.model.unique().tolist()
	model_name_map = {model : model_to_spec[model]['label'] for model in available_models}
	res['Model'] = res['model'].map(model_name_map)


	res = res[['Model', f'{FLAGS.metric} (std)_shift', f'{FLAGS.metric} (std)_no_shift']]
	print(res.to_latex(index=False))


if __name__ == '__main__':
	app.run(main)
