import sys
sys.path.append('/data/ddmg/slabs/structured_labels/')

import shared.utils as utils
import collections
import subprocess

if __name__=="__main__":
	bashCommand = f'tensorboard --port=6000 --logdir_spec ' 
	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},' 


	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'


	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'


	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "False",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		# "balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'


	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a0_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'


	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rbias_a1e3_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a1e3_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.8,
		'py1_y0': 0.95,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'bias_a0_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 0.0,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a0_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	param_dict = {
		'random_seed': 0,
		'pflip0': 0.01,
		'pflip1': 0.01,
		'py0': 0.5,
		'py1_y0': 0.5,
		'pixel': 64,
		'l2_penalty': 0.0,
		'dropout_rate': 0.0,
		'embedding_dim': 10,
		'sigma': 10.0,
		# 'alpha': 1e-3, 1e-1, 1, 1e1, 1e3,
		'alpha': 1e3,
		"architecture": "pretrained_resnet",
		"batch_size": 64,
		'weighted_mmd': "True",
		"balanced_weights": "False"
	}
	config = collections.OrderedDict(sorted(param_dict.items()))
	hash_string = utils.config_hasher(config)
	bashCommand = bashCommand + f'rct_a1e3_wb:/data/scratch/mmakar/waterbirds/tuning/{hash_string},'

	# bashCommand = (f'tensorboard --port=6000 --logdir_spec ' 
	# 	f'rct_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string1},'
	# 	f'rct_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string2},'
	# 	f'rct_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string8},'
	# 	f'rct_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string3},'
	# 	f'bias_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string4},'
	# 	f'bias_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string5},'
	# 	f'bias_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string6},'
	# 	f'bias_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string7},'
	# 	f'rbias_a1e3_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string9},'
	# 	f'rbias_a1e3_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string10},'
	# 	f'rbias_a0_u:/data/scratch/mmakar/waterbirds/tuning/{hash_string11},'
	# 	f'rbias_a0_w:/data/scratch/mmakar/waterbirds/tuning/{hash_string12}'
	# 	)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
