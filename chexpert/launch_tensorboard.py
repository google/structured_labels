""" Script for launching tensorboard for multiple networks """
import sys
sys.path.append('/data/ddmg/slabs/structured_labels/')

import shared.utils as utils
import collections
import subprocess
import shutil, glob

if __name__ == "__main__":
	move_old = False
	scratch_dir = '/data/scratch/mmakar/chexpert/tuning'
	bashCommand = 'tensorboard --port=6000 --logdir_spec '
	random_seed = 0

	for batch_size in [64, 128]:
		for pixel in [128, 256]:
			param_dict = {
				'random_seed': random_seed,
				'pixel': pixel,
				'l2_penalty': 0.0,
				'dropout_rate': 0.0,
				'embedding_dim': 10,
				'sigma': 10.0, 
				'alpha':  0.0, 
				"architecture": "pretrained_densenet",
				"batch_size": batch_size,
				'weighted_mmd': 'False',
				"balanced_weights": 'False',
				'minimize_logits': "False",
			}
			config = collections.OrderedDict(sorted(param_dict.items()))
			hash_string = utils.config_hasher(config)
			bashCommand = bashCommand + (
				f'2w_a{alpha}:{scratch_dir}/{hash_string},'
			)

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
