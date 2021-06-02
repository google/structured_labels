import os
import collections
import itertools
import functools
import pickle
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from shared import train_utils
import multiprocessing
import tqdm
tf.autograph.set_verbosity(0)



from waterbirds.data_builder import map_to_image_label, load_created_data
import shared.utils as utils
from shared import evaluation_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'waterbirds'))


def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	latest_model_dir = str(sorted(subdirs)[-1])
	loaded = tf.saved_model.load(latest_model_dir)
	model = loaded.signatures["serving_default"]
	return model


def get_data(random_seed, py0, py1_y0, pixel):
	experiment_directory = (f'/data/ddmg/slabs/waterbirds/experiment_data/'
		f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}')


	train_data, valid_data, shifted_data_dict = load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
	map_to_image_label_given_pixel = functools.partial(map_to_image_label,
		pixel=pixel)

	batch_size = len(valid_data)
	data_dict = {}
	# -- validation 
	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	data_dict['valid'] = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)


	for py in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]:
		shifted_test_data = shifted_data_dict[py]
		batch_size = len(shifted_test_data)
		eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
		eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
		eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
		data_dict[f'test_{py}'] = eval_shift_dataset
	
	return data_dict 


def get_recalculated_performance(config):
	# -- get the dataset 
	data_dict = get_data(config['random_seed'], 
		config['py0'], config['py1_y0'], config['pixel'])

	# -- get the hash directory where the model lives
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string, 'saved_model')

	# -- get model 
	model = get_last_saved_model(hash_dir)

	# -- set parameters for calculating the mmd 
	params = {
		'weighted_mmd': config['weighted_mmd'],
		'balanced_weights': config['balanced_weights'],
		'minimize_logits': config['minimize_logits'],
		'sigma': config['sigma'], 
		'alpha': config['alpha'], 
		'label_ind': 0}

	# -- get recalculated values
	recalculated_auc = {}
	for data in data_dict.items():
		data_name, dataset = data
		print(data_name)
		for batch_id, examples in enumerate(dataset):
			if batch_id > 0:
				raise ValueError("should be 0 only!")
			x, labels_weights = examples
			sample_weights, sample_weights_pos, sample_weights_neg = train_utils.extract_weights(
				labels_weights, params)
			labels = labels_weights['labels'].numpy()

			ypred =  model(tf.convert_to_tensor(x))['probabilities']
			ypred_trunc = ypred.numpy()
			ypred_trunc[ypred_trunc < 1e-3] = 1e-3
			ypred_trunc[ypred_trunc> 1 - 1e-3] = 1 - 1e-3
			# auc = roc_auc_score(labels[:,0], ypred.numpy())
			auc = np.mean(labels[:,0]* np.log(ypred_trunc) + (1- labels[:, 0])* np.log(1 - ypred_trunc))
			recalculated_auc[data_name] = auc
	
	# -- get old values 
	hash_dir = os.path.join(BASE_DIR, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')

	old_auc = pickle.load(open(performance_file, 'rb'))
	# old_auc = {k:v for k,v in old_auc.items() if 'auc' in k}
	old_auc = {k:v for k,v in old_auc.items() if 'loss' in k}

	print("==new===")
	print(recalculated_auc)
	print("==old==")
	print(old_auc)


if __name__=="__main__":
	param_dict = {
		'random_seed': [1],
		'pflip0': [0.01],
		'pflip1': [0.01],
		'py0': [0.5],
		'py1_y0': [0.5],
		'pixel': [128],
		'l2_penalty': [0.0],
		'dropout_rate': [0.0],
		'embedding_dim': [10],
		'sigma': [10.0],
		'alpha': [1000.0],
		"architecture": ["pretrained_resnet"],
		"batch_size": [64],
		'weighted_mmd': ['False'],
		"balanced_weights": ['False'],
		'minimize_logits': ["False"]
	}

	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	all_config = [dict(zip(keys, v)) for v in itertools.product(*values)]
	config = all_config[0]
	print(config)
	get_recalculated_performance(config)

