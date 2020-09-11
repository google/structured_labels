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

"""Creates config dictionaries for different experiments and models."""
import itertools
import collections


def cmnist_correlations_slabs():
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.05],
		'pflip1': [0.05],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1, 1.0, 10.0, 100.0],
		'alpha': [1.0, 10.0, 100.0, 1e5, 1e10]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep

def cmnist_correlations_opslabs():
	"""Creates hyperparameters for correlations experiment for
		overparameterized SLABS (our) model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.05],
		'pflip1': [0.05],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1, 1.0, 10.0, 100.0],
		'alpha': [1.0, 10.0, 100.0, 1e5, 1e10]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def cmnist_correlations_simple_baseline():
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.05],
		'pflip1': [0.05],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1],
		'alpha': [0.0]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def cmnist_correlations_oracle_aug(aug_prop):
	"""Creates hyperparameters for the correlations experiment for baseline.
	Args:
		aug_prop: float, proportion of training data to use for augmentation
	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.05],
		'pflip1': [0.05],
		'oracle_prop': [aug_prop],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1],
		'alpha': [0.0]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep
def cmnist_no_overlap_slabs():
	"""Creates hyperparameters for overlap experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.0],
		'pflip1': [0.0],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1, 1.0, 10.0, 100.0],
		'alpha': [1.0, 10.0, 100.0, 1e5, 1e10]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def cmnist_no_overlap_simple_baseline():
	"""Creates hyperparameters for overlap experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [int(i) for i in range(2)],
		'pflip0': [0.0],
		'pflip1': [0.0],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		'dropout_rate': [0.0, 0.01, 0.1],
		'embedding_dim': [10, 100, 1000],
		'py1_y0_s': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'sigma': [0.1],
		'alpha': [0.0]
	}

	param_dict = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
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
	if experiment not in ['correlation', 'overlap']:
		raise NotImplementedError((f'Experiment {experiment} parameter'
															' configuration not implemented'))
	if model not in ['slabs', 'opslabs', 'simple_baseline', 'oracle_aug']:
		raise NotImplementedError((f'Model {model} parameter configuration'
															' not implemented'))
	if model in ['oracle_aug'] and aug_prop <0.0:
		raise ValueError('Augmentation proportion is needed for augmentation'
											' baselines')

	if experiment == 'correlation' and model == 'slabs':
		return cmnist_correlations_slabs()
	if experiment == 'correlation' and model == 'opslabs':
		return cmnist_correlations_opslabs()
	if experiment == 'correlation' and model == 'simple_baseline':
		return cmnist_correlations_simple_baseline()
	if experiment == 'correlation' and model == 'oracle_aug':
		return cmnist_correlations_oracle_aug(aug_prop)
	if experiment == 'overlap' and model == 'slabs':
		return cmnist_no_overlap_slabs()
	if experiment == 'overlap' and model == 'simple_baseline':
		return cmnist_no_overlap_simple_baseline()
