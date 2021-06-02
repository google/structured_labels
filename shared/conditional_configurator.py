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

"""Creates configurations for SLAB models."""

import pandas as pd
import shared.cross_validation as cv


def get_sigma_foreach_lambda(configs, alphas, param_dict, base_dir):

	if 'alpha' in param_dict.keys():
		param_dict.pop('alpha', None)
	if 'sigma' in param_dict.keys():
		param_dict.pop('sigma', None)

	res_all = cv.import_results(configs, base_dir)

	lambda_combinations = res_all[['dropout_rate', 'l2_penalty',
		'embedding_dim']].copy()
	lambda_combinations.drop_duplicates(inplace=True)
	lambda_combinations.reset_index(inplace=True, drop=True)

	best_sigmas = []
	for i in range(lambda_combinations.shape[0]):
		lambda_vals = lambda_combinations.iloc[[i]].copy()

		# -- get all the validation results for this lambda
		res_lambda = res_all.merge(lambda_vals, on=lambda_vals.columns.tolist())

		# -- get best sigma for this lambda
		sigma_lambda = cv.get_optimal_sigma(res_lambda)
		lambda_vals['sigma'] = sigma_lambda
		best_sigmas.append(lambda_vals)

	# -- get all the data for lambda, alpha_lambda, sigma_lambda
	best_sigmas = pd.concat(best_sigmas, axis=0)
	best_sigmas['pflip0'] = param_dict['pflip0'][0]
	best_sigmas['pflip1'] = param_dict['pflip1'][0]
	best_sigmas['weighted_mmd'] = param_dict['weighted_mmd'][0]

	config_dataset = []
	for alpha in alphas:
		config_dataset_alpha = best_sigmas.copy()
		config_dataset_alpha['alpha'] = alpha
		for rs in param_dict['random_seed']:
			config_dataset_alpha_i = config_dataset_alpha.copy()
			config_dataset_alpha_i['random_seed'] = rs
			config_dataset.append(config_dataset_alpha_i)

	config_dataset = pd.concat(config_dataset, axis=0)
	print(config_dataset)

	return config_dataset.to_dict('records')

