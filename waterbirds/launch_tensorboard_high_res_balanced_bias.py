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
	minimize_logits = False
	# random_seed = 1

	for random_seed in [0, 1]:
		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 0.0,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "False",
			'balanced_weights': 'False', 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_u_a0_l0:{scratch_dir}/{hash_string},'

		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 1,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "False",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_u_a1_l0:{scratch_dir}/{hash_string},'


		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 1e3,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "False",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_u_a1e3_l0:{scratch_dir}/{hash_string},'

		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0001,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 0.0,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "False",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_u_a0_l1e-3:{scratch_dir}/{hash_string},'

		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 0.0,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "True",
			'balanced_weights': 'False', 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_v_a0_l0:{scratch_dir}/{hash_string},'

		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 1,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "True",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_v_a1_l0:{scratch_dir}/{hash_string},'


		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 1e3,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "True",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_v_a1e3_l0:{scratch_dir}/{hash_string},'

		if random_seed == 0:
			bashCommand = bashCommand + f'OLD_rs{random_seed}_v_a1e3_l0:{scratch_dir}/{hash_string}/old_mmd,'


		param_dict = {
			'random_seed': random_seed,
			'pflip0': 0.01,
			'pflip1': 0.01,
			'py0': 0.5,
			'py1_y0': 0.95,
			'pixel': 224,
			'l2_penalty': 0.0,
			'dropout_rate': 0.0,
			'embedding_dim': 10,
			'sigma': 10.0,
			'alpha': 1e4,
			"architecture": "pretrained_resnet",
			"batch_size": 64,
			'weighted_mmd': "True",
			"balanced_weights": "False", 
			'minimize_logits': minimize_logits
		}
		config = collections.OrderedDict(sorted(param_dict.items()))
		hash_string = utils.config_hasher(config)
		if move_old:
			print(hash_string)
			shutil.move(f'{scratch_dir}/{hash_string}/',
					f'{scratch_dir}/old_{hash_string}/')
			# shutil.rmtree(f'{results_dir}/{hash_string}/epoch_100_saved_model/')
			# shutil.move(f'{results_dir}/{hash_string}/performance.pkl',
			# 			f'{results_dir}/{hash_string}/epoch_100_performance.pkl')
			# shutil.move(f'{results_dir}/{hash_string}/saved_model/',
			# 			f'{results_dir}/{hash_string}/epoch_100_saved_model/')
		bashCommand = bashCommand + f'rs{random_seed}_v_a1e4_l0:{scratch_dir}/{hash_string},'

		if random_seed ==1:
			param_dict = {
				'random_seed': random_seed,
				'pflip0': 0.01,
				'pflip1': 0.01,
				'py0': 0.5,
				'py1_y0': 0.95,
				'pixel': 224,
				'l2_penalty': 0.0,
				'dropout_rate': 0.0,
				'embedding_dim': 10,
				'sigma': 1.0,
				'alpha': 1,
				"architecture": "pretrained_resnet",
				"batch_size": 64,
				'weighted_mmd': "True",
				"balanced_weights": "False", 
				'minimize_logits': minimize_logits
			}
			config = collections.OrderedDict(sorted(param_dict.items()))
			hash_string = utils.config_hasher(config)
			bashCommand = bashCommand + f'rs{random_seed}_v_a1_l0_sig1:{scratch_dir}/{hash_string},'


			param_dict = {
				'random_seed': random_seed,
				'pflip0': 0.01,
				'pflip1': 0.01,
				'py0': 0.5,
				'py1_y0': 0.95,
				'pixel': 224,
				'l2_penalty': 0.0,
				'dropout_rate': 0.0,
				'embedding_dim': 10,
				'sigma': 1.0,
				'alpha': 1e3,
				"architecture": "pretrained_resnet",
				"batch_size": 64,
				'weighted_mmd': "True",
				"balanced_weights": "False", 
				'minimize_logits': minimize_logits
			}
			config = collections.OrderedDict(sorted(param_dict.items()))
			hash_string = utils.config_hasher(config)

			bashCommand = bashCommand + f'rs{random_seed}_v_a1e3_l0_sig1:{scratch_dir}/{hash_string},'

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
