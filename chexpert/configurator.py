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

"""Creates config dictionaries for different experiments and models waterbirds"""
import os
import collections
import itertools
import re


def configure_slabs(skew_train, weighted_mmd, balanced_weights):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [i for i in range(5)],
		'pixel': [512],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		# 'sigma': [1.0, 10.0, 1e2, 1e3],
		# 'alpha': [1e3, 1e5, 1e7],
		'sigma': [1e2],
		'alpha': [1e5],
		"architecture": ["pretrained_densenet"],
		"batch_size": [16],
		'weighted_mmd': [weighted_mmd],
		"balanced_weights": [balanced_weights],
		'minimize_logits': ["False"],
		"skew_train": [skew_train], 
		'num_epochs': [3]
	}

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def configure_simple_baseline(skew_train, weighted):
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""

	param_dict = {
		'random_seed': [0],
		'pixel': [512],
		'l2_penalty': [0.0],
		# 'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_densenet"],
		"batch_size": [16],
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
		"skew_train": ['True'], 
		"num_epochs": [3]
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(param_dict)

	return sweep


def configure_random_augmentation(skew_train, weighted):
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""

	param_dict = {
		'random_seed': [i for i in range(10)],
		'pixel': [128],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_densenet"],
		"batch_size": [64],
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
		"random_augmentation": ['True'],
		"skew_train": [skew_train]
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(param_dict)

	return sweep



def get_sweep(experiment, model):
	"""Wrapper function, creates configurations based on experiment and model.

	Args:
		experiment: string with experiment name
		model: string, which model to create the configs for
		aug_prop: float, proportion of augmentation relative to training data.
			Only relevant for augmentation based baselines

	Returns:
		Iterator with all hyperparameter combinations
	"""
	implemented_models = [
		'slabs_weighted', 'slabs_weighted_bal', 'slabs_weighted_bal_two_way',
		'slabs_warmstart_weighted', 'slabs_warmstart_weighted_bal',
		'slabs_logit',
		'unweighted_slabs', 'unweighted_slabs_logit',
		'simple_baseline','weighted_baseline',
		'random_aug', 'weighted_random_aug']

	implemented_experiments = ['skew_train', 'unskew_train']


	if experiment not in implemented_experiments:
		raise NotImplementedError((f'Experiment {experiment} parameter'
															' configuration not implemented'))
	if model not in implemented_models:
		raise NotImplementedError((f'Model {model} parameter configuration'
															' not implemented'))


	skew_train = 'True' if experiment == 'skew_train' else 'False'



	if model == 'slabs_weighted':
		return configure_slabs(skew_train=skew_train,
			weighted_mmd='True', balanced_weights='False')

	# if model == 'slabs_warmstart_weighted':
	# 	return configure_slabs(py0, py1_y0, logit='False',
	# 		weighted_mmd='True', balanced_weights= 'False', two_way_mmd=False,
	# 		warmstart=True, asym_train=asym_train, clean_back=clean_back)

	if model == 'slabs_weighted_bal':
		return configure_slabs(skew_train=skew_train,
			weighted_mmd='True', balanced_weights='True')

	# if model == 'slabs_weighted_bal_two_way':
	# 	return configure_slabs(py0, py1_y0, logit='False',
	# 		weighted_mmd='True', balanced_weights='True', two_way_mmd=True,
	# 		warmstart=False, asym_train=asym_train, clean_back=clean_back)

	# if model == 'slabs_warmstart_weighted_bal':
	# 	return configure_slabs(py0, py1_y0, logit='False', weighted_mmd='True',
	# 		balanced_weights= 'True', two_way_mmd=False,
	# 		warmstart=True, asym_train=asym_train, clean_back=clean_back)


	# if model == 'slabs_logit':
	# 	return configure_slabs(py0, py1_y0, logit='True', weighted_mmd='True',
	# 		balanced_weights = 'True', two_way_mmd=False,
	# 		warmstart=False, asym_train=asym_train, clean_back=clean_back)

	if model == 'unweighted_slabs':
		return configure_slabs(skew_train=skew_train,
			weighted_mmd='False', balanced_weights='False')

	# if model == 'unweighted_slabs_logit':
	# 	return configure_slabs(py0, py1_y0, logit='True',
	# 		weighted_mmd='False', balanced_weights='False', two_way_mmd=False,
	# 		warmstart=False, asym_train=asym_train, clean_back=clean_back)

	if model == 'simple_baseline':
		return configure_simple_baseline(skew_train=skew_train, weighted='False')

	if model == 'weighted_baseline':
		return configure_simple_baseline(skew_train=skew_train, weighted='True')

	if model == 'random_aug':
		return configure_random_augmentation(skew_train=skew_train, weighted='False')

	# if model == 'weighted_random_aug':
	# 	return configure_random_augmentation(py0, py1_y0, weighted='True',
	# 		asym_train=asym_train, clean_back=clean_back)


