""" Script for launching tensorboard for multiple networks """
import sys
sys.path.append('/data/ddmg/slabs/structured_labels/')

import shared.utils as utils
import collections
import subprocess

if __name__ == "__main__":
	results_dir = '/data/scratch/mmakar/waterbirds/tuning'
	bashCommand = 'tensorboard --port=6000 --logdir_spec '

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 224,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False", 
		'balanced_weights': 'False'
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'a0_l0:{results_dir}/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 224,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		'alpha': 1,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'a1_l0:{results_dir}/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 224,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'a1e3_l0:{results_dir}/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 224,
		'l2_penalty': 0.0001,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'a0_l1e-3:{results_dir}/{hash_string}'

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
