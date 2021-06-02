import functools
import itertools
import subprocess
import multiprocessing
import os
import pickle

import argparse
import numpy as np  # noqa: F401
import tqdm

from shared.utils import config_hasher, tried_config
from cmnist import configurator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))

if __name__ == "__main__":
	experiment_name = 'correlation'
	model_to_tune = 'twopslabs'
	aug_prop = 1.1

	all_config = configurator.get_sweep(experiment_name, model_to_tune, aug_prop)
	config = all_config[0]
	print(config)
	hash_string = config_hasher(config)
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	if not os.path.exists(hash_dir):
		os.system(f'mkdir -p {hash_dir}')
	config['exp_dir'] = hash_dir
	config['cleanup'] = True
	# chosen_gpu = get_gpu_assignment()
	chosen_gpu = '0'
	config['gpuid'] = chosen_gpu
	flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	subprocess.call('python -m cmnist.main %s' % flags,
		shell=True)