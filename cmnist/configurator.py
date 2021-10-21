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
import os
import collections
import itertools
import re


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))

def configure_slabs(py0, py1_y0, logit, weighted_mmd, balanced_weights,
	two_way_mmd, warmstart, batch_size=1e10):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	if batch_size==1e10:
		batch_size = [16, 32, 64, 128, 256]
	else:
		batch_size = [int(batch_size)]
	param_dict = {
		'random_seed': [0],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [20],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'sigma': [0.1, 1.0, 10.0, 100.0, 1000.0],
		'alpha': [0.1, 1.0, 10.0, 100.0, 1e5, 1e7],
		"architecture": ["simple"],
		"batch_size": batch_size,
		'weighted_mmd': [weighted_mmd],
		"balanced_weights": [balanced_weights],
		'minimize_logits': ["False"],
	}

	if warmstart:
		param_dict['warmstart_dir'] = ['find']
	if two_way_mmd:
		param_dict['two_way_mmd'] = ['True']

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_simple_baseline(py0, py1_y0, weighted, batch_size=1e10):
	"""Creates hyperparameters for the correlations experiment for baseline.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	if batch_size==1e10:
		batch_size = [16, 32, 64, 128, 256]
	else:
		batch_size = [int(batch_size)]
	param_dict = {
		'random_seed': [0],
		'pflip0': [0.05],
		'pflip1': [0.05],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [20],
		'l2_penalty': [0.0, 0.1, 1.0, 10.0, 100.0],
		# 'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["simple"],
		"batch_size": batch_size,
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(param_dict)

	return sweep

def configure_rex(py0, py1_y0, weighted):
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
		'pixel': [20],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'sigma': [10.0],
		'alpha': [10**6],
		"architecture": ["simple"],
		"batch_size": [32],
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
		'rex': [ 'True_norm']
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(param_dict)

	return sweep

def configure_random_augmentation(py0, py1_y0, weighted):
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
		'pixel': [20],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["simple"],
		"batch_size": [32],
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
		"random_augmentation": ['True']
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(param_dict)

	return sweep


def configure_oracle_augmentation(py0, py1_y0, oracle_prop, weighted):
	"""Creates hyperparameters for the correlations experiment for baseline.
	Args:
		aug_prop: float, proportion of training data to use for augmentation
	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [i for i in range(10)],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [py0],
		'py1_y0': [py1_y0],
		'pixel': [20],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [1000],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["simple"],
		"batch_size": [32],
		'weighted_mmd': [weighted],
		"balanced_weights": [weighted],
		'minimize_logits': ["False"],
		"oracle_prop": [oracle_prop],
	}

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]
	return sweep


def get_sweep(experiment, model, batch_size, oracle_prop=-1.0):
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
		'slabs_logit', 'slabs_unweighted_two_way',
		'unweighted_slabs', 'unweighted_slabs_logit',
		'simple_baseline','weighted_baseline',
		'oracle_aug', 'weighted_oracle_aug',
		'random_aug', 'weighted_random_aug', 'rex']

	implemented_experiments = ['5090', '5095']

	if experiment not in implemented_experiments:
		raise NotImplementedError((f'Experiment {experiment} parameter'
															' configuration not implemented'))
	if model not in implemented_models:
		raise NotImplementedError((f'Model {model} parameter configuration'
															' not implemented'))
	if model in ['oracle_aug'] and oracle_prop < 0.0:
		raise ValueError('Augmentation proportion is needed for augmentation'
											' baselines')

	py0 = float(experiment[:2]) / 100.0
	py1_y0 = float(experiment[2:]) / 100.0


	if model == 'slabs_weighted':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='True', balanced_weights='False', two_way_mmd=False,
			warmstart=False)

	if model == 'slabs_warmstart_weighted':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='True', balanced_weights= 'False', two_way_mmd=False,
			warmstart=True)

	if model == 'slabs_weighted_bal':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='True', balanced_weights='True', two_way_mmd=False,
			warmstart=False)

	if model == 'slabs_weighted_bal_two_way':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='True', balanced_weights='True', two_way_mmd=True,
			warmstart=False)

	if model == 'slabs_unweighted_two_way':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='False', balanced_weights='False', two_way_mmd=True,
			warmstart=False)


	if model == 'slabs_warmstart_weighted_bal':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='True', balanced_weights= 'True', two_way_mmd=False,
			warmstart=True)


	if model == 'slabs_logit':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='True',
			weighted_mmd='True', balanced_weights='True', two_way_mmd=False,
			warmstart=False)

	if model == 'unweighted_slabs':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='False',
			weighted_mmd='False', balanced_weights='False', two_way_mmd=False,
			warmstart=False,)

	if model == 'unweighted_slabs_logit':
		return configure_slabs(py0, py1_y0, batch_size=batch_size, logit='True',
			weighted_mmd='False', balanced_weights='False', two_way_mmd=False,
			warmstart=False)

	if model == 'simple_baseline':
		return configure_simple_baseline(py0, py1_y0, weighted='False', batch_size=batch_size)

	if model == 'weighted_baseline':
		return configure_simple_baseline(py0, py1_y0, weighted='True')

	if model == 'rex':
		return configure_rex(py0, py1_y0, weighted='False')

	if model == 'random_aug':
		return configure_random_augmentation(py0, py1_y0, weighted='False')

	if model == 'weighted_random_aug':
		return configure_random_augmentation(py0, py1_y0, weighted='True')

	if model == 'oracle_aug':
		return configure_oracle_augmentation(py0, py1_y0, oracle_prop,
			weighted='False')

	if model == 'weighted_oracle_aug':
		return configure_oracle_augmentation(py0, py1_y0, oracle_prop,
			weighted='True')

