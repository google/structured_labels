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

"""Runs the full correlation sweep for the corrupted mnist experiment."""

import hashlib
from subprocess import call
from multiprocessing import Pool
import os
import pickle
import tqdm
from cmnist import configurator

N_TRIALS = 10
EXP_NAME = 'correlation'
MODEL_TO_TUNE = 'simple_baseline'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist'))
NUM_WORKERS = 5
OVERRIDE = True

def runner(config):
	"""Trains model in config if not trained before.
	Args:
		config: dict with config
	Returns:
		Nothing
	"""
	config_string = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	hash_string = hashlib.sha256(config_string.encode()).hexdigest()
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')
	if os.path.isfile(performance_file) and not OVERRIDE:
		print("Tried this config, skipping")
		return None
	if not os.path.exists(hash_dir):
		os.system(f'mkdir -p {hash_dir}')
	config['exp_dir'] = hash_dir
	flags = ' '.join('--%s %s' % (k, str(v)) for k, v in config.items())
	call('python -m cmnist.main %s > /dev/null 2>&1' % flags, shell=True)
	# call('python -m cmnist.main %s' % flags, shell=True)
	config.pop('exp_dir')
	pickle.dump(config, open(os.path.join(hash_dir, 'config.pkl'), 'wb'))


if __name__ == '__main__':

	all_config = configurator.get_sweep(EXP_NAME, MODEL_TO_TUNE)

	# TODO: random sample instead of first N
	all_config = all_config[:N_TRIALS]

	if NUM_WORKERS > 1:
		pool = Pool(NUM_WORKERS)
		for _ in tqdm.tqdm(pool.imap_unordered(runner, all_config),
			total=len(all_config)):
			pass
	else:
		for config in all_config:
			runner(config)
