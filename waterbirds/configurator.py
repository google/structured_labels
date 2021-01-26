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


def configure_slabs(py0, py1_y0, logit):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [i for i in range(10)],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [128],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [1.0, 10.0, 100.0, 1000.0],
		'alpha': [1e3],
		"architecture": ["pretrained_resnet"],
		"batch_size": [64],
		'weighted_mmd': ["True"],
		"balanced_weights": ["True"],
		'minimize_logits': [logit]
	}

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def configure_unweighted_slabs(py0, py1_y0, logit):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [i for i in range(10)],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [128],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [1.0, 10.0, 100.0, 1000.0],
		'alpha': [1e3],
		"architecture": ["pretrained_resnet"],
		"batch_size": [64],
		'weighted_mmd': ["False"],
		"balanced_weights": ["False"],
		'minimize_logits': [logit]
	}

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_simple_baseline(py0, py1_y0):
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""

	param_dict = {
		'random_seed': [i for i in range(10)],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [128],
		'l2_penalty': [0.0, 0.0001],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_resnet"],
		"batch_size": [64],
		'weighted_mmd': ["False"],
		"balanced_weights": ["False"],
		'minimize_logits': ["False"]
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_weighted_baseline(py0, py1_y0):
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""

	param_dict = {
		'random_seed': [i for i in range(10)],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [128],
		'l2_penalty': [0.0, 0.0001],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_resnet"],
		"batch_size": [64],
		'weighted_mmd': ["True"],
		"balanced_weights": ["True"],
		'minimize_logits': ["False"]
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def configure_oracle_aug(aug_prop):
	"""Creates hyperparameters for the correlations experiment for baseline.
	Args:
		aug_prop: float, proportion of training data to use for augmentation
	Returns:
		Iterator with all hyperparameter combinations
	"""
	raise NotImplementedError("not yet")
	param_dict = {
		'random_seed': [0],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py1_y0_shift_list': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'oracle_prop': [aug_prop],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.1],
		'embedding_dim': [10, 100, 1000],
		'sigma': [0.1],
		'alpha': [0.0]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep1 = [dict(zip(keys, v)) for v in itertools.product(*values)]

	param_dict = {
		'random_seed': [0],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py1_y0': [1.0],
		'py1_y0_shift_list': [0.2, 0.4, 0.6, 0.8, 1],
		'oracle_prop': [aug_prop],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'sigma': [0.1],
		'alpha': [0.0]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep2 = [dict(zip(keys, v)) for v in itertools.product(*values)]
	sweep = sweep1
	return sweep


def get_sweep(experiment, model, aug_prop=-1.0):
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
		'slabs', 'slabs_logit',
		'unweighted_slabs', 'unweighted_slabs_logit',
		'simple_baseline','weighted_baseline',
		'oracle_aug']

	implemented_experiments = ['5090', '5050', '8090']

	if model[:10] == "oracle_aug" and len(model) > 10:
		match = re.match(r'.*(\_)', model)
		start_pos = match.end(1)
		aug_prop = float(model[start_pos:])
		model = model[:10]

	if experiment not in implemented_experiments:
		raise NotImplementedError((f'Experiment {experiment} parameter'
															' configuration not implemented'))
	if model not in implemented_models:
		raise NotImplementedError((f'Model {model} parameter configuration'
															' not implemented'))
	if model in ['oracle_aug'] and aug_prop < 0.0:
		raise ValueError('Augmentation proportion is needed for augmentation'
											' baselines')

	py0 = float(experiment[:2]) / 100.0
	py1_y0 = float(experiment[2:]) / 100.0

	if model == 'slabs':
		return configure_slabs(py0, py1_y0, logit='False')

	if model == 'slabs_logit':
		return configure_slabs(py0, py1_y0, logit='True')

	if model == 'unweighted_slabs':
		return configure_unweighted_slabs(py0, py1_y0, logit='False')

	if model == 'unweighted_slabs_logit':
		return configure_unweighted_slabs(py0, py1_y0, logit='True')

	if model == 'simple_baseline':
		return configure_simple_baseline(py0, py1_y0)

	if model == 'weighted_baseline':
		return configure_weighted_baseline(py0, py1_y0)

