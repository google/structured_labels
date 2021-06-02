""" Script for launching tensorboard for multiple networks """
import sys
sys.path.append('/data/ddmg/slabs/structured_labels/')

import shared.utils as utils
import collections
import subprocess
import shutil, glob

if __name__ == "__main__":
	move_old = False
	scratch_dir = '/data/scratch/mmakar/waterbirds/tuning'
	results_dir = '/data/ddmg/slabs/waterbirds/tuning'
	bashCommand = 'tensorboard --port=6000 --logdir_spec '
	# random_seed = 1

	for alpha in [1e3, 1e5, 1e7]:
		py0 = 0.8
		py1_y0 = 0.9

		param_dict = {
			'random_seed': 22,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': py0,
			'py1_y0': py1_y0,
			'pixel': 128,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 0.1, 
			'alpha': alpha, 
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': 'True',
			"balanced_weights": 'True',
			'minimize_logits': "False",
			'clean_back': 'True', 
			'two_way_mmd': 'True'
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		bashCommand = bashCommand + (
			f'2w_a{alpha}:{scratch_dir}/{hash_string},'
		)

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
