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

"""Functions to create the chexpert datasets."""
import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = '/data/ddmg/slabs/chexpert'
MAIN_DIR = '/data/ddmg/slabs'

FEATURE_DESCRIPTION = {
    'image_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'y0': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'y1': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    
    'unbalanced_weights_pos': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'unbalanced_weights_neg': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'unbalanced_weights_both': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    
    'balanced_weights_pos': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'balanced_weights_neg': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'balanced_weights_both': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    
}


def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img

def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label



def map_to_image_label_ser(example_proto, pixel):
		# Parse the input `tf.train.Example` proto using the dictionary above.
		parsed_dict =  tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
		# resize, rescale  image
		img = tf.io.read_file(parsed_dict['image_path'])
		img = tf.image.decode_jpeg(img, channels=3)
		
		img = tf.image.resize(img, (pixel, pixel))
		img = img / 255
		
		labels = tf.concat([
			tf.reshape(parsed_dict['y0'], (1, 1)), 
			tf.reshape(parsed_dict['y1'], (1, 1))], axis = 0)    
		unbalanced_weights = tf.concat([ 
			tf.reshape(parsed_dict['unbalanced_weights_pos'], (1, 1)), 
			tf.reshape(parsed_dict['unbalanced_weights_neg'], (1, 1)), 
			tf.reshape(parsed_dict['unbalanced_weights_both'], (1, 1))], 
			axis = 0 )
		
		balanced_weights = tf.stack([ 
			tf.reshape(parsed_dict['balanced_weights_pos'], (1, 1)), 
			tf.reshape(parsed_dict['balanced_weights_neg'], (1, 1)), 
			tf.reshape(parsed_dict['balanced_weights_both'], (1, 1))], 
			axis = 0 )
		labels_and_weights = {
				'labels': labels,
				'unbalanced_weights': unbalanced_weights,
				'balanced_weights': balanced_weights,
		}
				
		return img, labels_and_weights


def map_to_image_label(x, pixel):

	weights_included = len(x) > 3

	chest_image = x[0]
	y0 = x[1]
	y1 = x[2]

	if weights_included:
		unbalanced_weights_pos = x[3]
		unbalanced_weights_neg = x[4]
		unbalanced_weights_both = x[5]

		balanced_weights_pos = x[6]
		balanced_weights_neg = x[7]
		balanced_weights_both = x[8]

	# decode images
	img = read_decode_jpg(chest_image)

	# resize, rescale  image
	img = tf.image.resize(img, (pixel, pixel))
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	labels = tf.concat([y0, y1], axis=0)

	# get the weights
	if weights_included:
		unbalanced_weights_pos = decode_number(unbalanced_weights_pos)
		unbalanced_weights_neg = decode_number(unbalanced_weights_neg)
		unbalanced_weights_both = decode_number(unbalanced_weights_both)

		unbalanced_weights = tf.concat([unbalanced_weights_pos,
			unbalanced_weights_neg, unbalanced_weights_both], axis=0)

		balanced_weights_pos = decode_number(balanced_weights_pos)
		balanced_weights_neg = decode_number(balanced_weights_neg)
		balanced_weights_both = decode_number(balanced_weights_both)

		balanced_weights = tf.concat([balanced_weights_pos,
			balanced_weights_neg, balanced_weights_both], axis=0)

	else:
		unbalanced_weights = None
		balanced_weights = None

	labels_and_weights = {
		'labels': labels,
		'unbalanced_weights': unbalanced_weights,
		'balanced_weights': balanced_weights,
	}

	return img, labels_and_weights


def get_weights(data_frame):

	male_pnu_weight = np.sum(
		data_frame.y0 * data_frame.y1) / np.sum(data_frame.y1)
	male_hlthy_weight = np.sum(
		(1.0 - data_frame.y0) * data_frame.y1) / np.sum(data_frame.y1)

	female_pnu_weight = np.sum(
		data_frame.y0 * (1.0 - data_frame.y1)) / np.sum((1.0 - data_frame.y1))
	female_hlthy_weight = np.sum(
		(1.0 - data_frame.y0) * (1.0 - data_frame.y1)) / np.sum(
		(1.0 - data_frame.y1))

	# -- positive weights
	data_frame['weights_pos'] = data_frame.y0 * male_pnu_weight + \
		(1.0 - data_frame.y0) * male_hlthy_weight
	data_frame['weights_pos'] = 1.0 / data_frame['weights_pos']
	data_frame['weights_pos'] = data_frame.y1 * data_frame['weights_pos']

	assert data_frame.weights_pos.isin([np.nan, np.inf, -np.inf]).sum() == 0

	data_frame['balanced_weights_pos'] = np.mean(data_frame.y0) * \
		data_frame.y0 * data_frame.weights_pos + \
		np.mean(1.0 - data_frame.y0) * (1.0 - data_frame.y0) * data_frame.weights_pos

	# -- negative weights
	data_frame['weights_neg'] = data_frame.y0 * female_pnu_weight + \
		(1.0 - data_frame.y0) * female_hlthy_weight
	data_frame['weights_neg'] = 1.0 / data_frame['weights_neg']
	data_frame['weights_neg'] = (1.0 - data_frame.y1) * data_frame['weights_neg']

	assert data_frame.weights_neg.isin([np.nan, np.inf, -np.inf]).sum() == 0

	data_frame['balanced_weights_neg'] = np.mean(data_frame.y0) * \
		data_frame.y0 * data_frame.weights_neg + \
		np.mean(1.0 - data_frame.y0) * (1.0 - data_frame.y0) * data_frame.weights_neg

	# aggregate weights
	data_frame['weights'] = data_frame['weights_pos'] + data_frame['weights_neg']
	data_frame['balanced_weights'] = data_frame['balanced_weights_pos'] + \
		data_frame['balanced_weights_neg']

	return data_frame


def get_skewed_data(cand_df, rng):
		if rng is None:
				rng = np.random.RandomState(0)
				
		ids_11 = cand_df.uid[((cand_df.y1==1) &  (cand_df.y0==1))].tolist()
		ids_10_size = int(0.1 * (len(ids_11)/0.9))
		ids_10_candidates = cand_df.uid[((cand_df.y1==1) &  (cand_df.y0==0))].tolist()
		ids_10 = rng.choice(ids_10_candidates, size = ids_10_size, replace =False).tolist()

		ids_01 = cand_df.uid[((cand_df.y1==0) &  (cand_df.y0==1))].tolist()
		ids_00_size = int(0.9 * (len(ids_01)/0.1))
		ids_00_candidates = cand_df.uid[((cand_df.y1==0) &  (cand_df.y0==0))].tolist()
		ids_00 = rng.choice(ids_00_candidates, size = ids_00_size, replace =False).tolist()

		ids = ids_11 + ids_10 + ids_00 + ids_01
		sk_df = cand_df[(cand_df.uid.isin(ids))]
		return sk_df

def get_unskewed_data(cand_df, skewed_data, rng):
		if rng is None:
				rng = np.random.RandomState(0)

		n_y1 = skewed_data[(skewed_data.y0 == 1)].shape[0]
		n_y0 = skewed_data[(skewed_data.y0 == 0)].shape[0]

		ids_0 = rng.choice(
				cand_df.uid[(cand_df.y0==0)], 
				size = n_y0, replace=False).tolist() 
		ids_1 = rng.choice(
				cand_df.uid[(cand_df.y0==1)], 
				size = n_y1, replace=False).tolist() 

		ids = ids_0 + ids_1 
		usk_df = cand_df[(cand_df.uid.isin(ids))]
		return usk_df
		


def save_created_data(data_frame, experiment_directory, filename):
	txt_df = f'{MAIN_DIR}/' + data_frame.Path + \
		',' + data_frame.y0.astype(str) + \
		',' + data_frame.y1.astype(str) + \
		',' + data_frame.weights_pos.astype(str) + \
		',' + data_frame.weights_neg.astype(str) + \
		',' + data_frame.weights.astype(str) + \
		',' + data_frame.balanced_weights_pos.astype(str) + \
		',' + data_frame.balanced_weights_neg.astype(str) + \
		',' + data_frame.balanced_weights.astype(str)

	txt_df.to_csv(f'{experiment_directory}/{filename}.txt',
		index=False)


def load_created_data(experiment_directory, skew_train):

	skew_str = 'skew' if skew_train == 'True' else 'unskew'

	train_data = pd.read_csv(
		f'{experiment_directory}/{skew_str}_train.txt').values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = pd.read_csv(
		f'{experiment_directory}/{skew_str}_valid.txt').values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	test_data_dict = {}
	for skew_str in ['skew', 'unskew']:
		pskew = 0.9 if skew_str == 'skew' else 0.5
		test_data = pd.read_csv(
			f'{experiment_directory}/{skew_str}_test.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		test_data_dict[pskew] = test_data

	return train_data, validation_data, test_data_dict



def create_save_chexpert_lists(experiment_directory, p_tr=.7, random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	# --- read in the cleaned image filenames (see chexpert_creation)
	df = pd.read_csv(f'{DATA_DIR}/clean_data.csv')

	# ---- split into train and test patients
	tr_val_candidates = rng.choice(df.patient.unique(), 
		size = int(len(df.patient.unique())*p_tr), replace = False).tolist()
	ts_candidates = list(set(df.patient.unique()) - set(tr_val_candidates))

	# --- split training into training and validation 
	# TODO: don't hard code the validation percent
	tr_candidates = rng.choice(tr_val_candidates,
		size=int(0.75 * len(tr_val_candidates)), replace=False).tolist()
	val_candidates = list(set(tr_val_candidates) - set(tr_candidates))

	tr_candidates_df = df[(df.patient.isin(tr_candidates))]
	val_candidates_df = df[(df.patient.isin(val_candidates))]
	ts_candidates_df = df[(df.patient.isin(ts_candidates))]

	# --- checks 
	assert len(ts_candidates) + len(tr_candidates) + len(val_candidates) == len(df.patient.unique())
	assert len(set(ts_candidates) & set(tr_candidates)) == 0
	assert len(set(ts_candidates) & set(val_candidates)) == 0
	assert len(set(tr_candidates) & set(val_candidates)) == 0

	# --- get train datasets
	tr_sk_df = get_skewed_data(tr_candidates_df, rng)
	tr_sk_df = get_weights(tr_sk_df)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
		filename='skew_train')

	tr_usk_df = get_unskewed_data(tr_candidates_df, tr_sk_df, rng)
	tr_usk_df = get_weights(tr_usk_df)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
		filename='unskew_train')


	# --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, rng)
	val_sk_df = get_weights(val_sk_df)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
		filename='skew_valid')

	val_usk_df = get_unskewed_data(val_candidates_df, val_sk_df, rng)
	val_usk_df = get_weights(val_usk_df)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
		filename='unskew_valid')


	# --- get test
	ts_sk_df = get_skewed_data(ts_candidates_df, rng)
	ts_sk_df = get_weights(ts_sk_df)
	save_created_data(ts_sk_df, experiment_directory=experiment_directory,
		filename='skew_test')

	ts_usk_df = get_unskewed_data(ts_candidates_df, ts_sk_df, rng)
	ts_usk_df = get_weights(ts_usk_df)
	save_created_data(ts_usk_df, experiment_directory=experiment_directory,
		filename='unskew_test')




def build_input_fns_sr(skew_train='False', p_tr=.7, Kfolds=0, random_seed=None):

	experiment_directory = (f'{DATA_DIR}/experiment_data/'
		f'rs{random_seed}')

	skew_str = 'unskew' if skew_train == 'False' else 'skew'

	# --load splits
	train_data_list, _, _ = load_created_data(
		experiment_directory=experiment_directory, skew_train=skew_train)

	# --this helps auto-set training steps at train time
	training_data_size = len(train_data_list)

	# Build an iterator over training batches.
	def train_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.TFRecordDataset(f'{experiment_directory}/{skew_str}_train.tfrecord')
		dataset = dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
		# dataset = dataset.shuffle(int(training_data_size * 0.05)).batch(batch_size).repeat(num_epochs)
		dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.TFRecordDataset(f'{experiment_directory}/{skew_str}_valid.tfrecord')
		valid_dataset = valid_dataset.map(map_to_image_label_given_pixel,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
		return valid_dataset


	Kfold_input_fn_creater = None

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, asym=False):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		if py == 0.5:
			def eval_input_fn():
				eval_shift_dataset = tf.data.TFRecordDataset(f'{experiment_directory}/unskew_test.tfrecord')
				eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
				eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
				return eval_shift_dataset
		else: 
			def eval_input_fn():
				eval_shift_dataset = tf.data.TFRecordDataset(f'{experiment_directory}/skew_test.tfrecord')
				eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
				eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
				return eval_shift_dataset

		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater


def build_input_fns(skew_train='False', p_tr=.7, Kfolds=0, random_seed=None):

	experiment_directory = (f'{DATA_DIR}/experiment_data/'
		f'rs{random_seed}')

	if not os.path.exists(experiment_directory):
		os.mkdir(experiment_directory)

	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/sk_train.txt'):

		create_save_chexpert_lists(
			experiment_directory=experiment_directory,
			p_tr=p_tr,
			random_seed=random_seed)

	# --load splits
	train_data, valid_data, shifted_data_dict = load_created_data(
		experiment_directory=experiment_directory, skew_train=skew_train)

	# --this helps auto-set training steps at train time
	training_data_size = len(train_data)

	# Build an iterator over training batches.
	def train_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
		# dataset = dataset.shuffle(int(training_data_size * 0.05)).batch(batch_size).repeat(num_epochs)
		dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label_given_pixel,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
		return valid_dataset

	# -- Create kfold splits
	if Kfolds > 0:
		effective_validation_size = int(int(len(valid_data) / Kfolds) * Kfolds)
		batch_size = int(effective_validation_size / Kfolds)

		valid_splits = np.random.choice(len(valid_data),
			size=effective_validation_size, replace=False).tolist()

		valid_splits = [
			valid_splits[i:i + batch_size] for i in range(0, effective_validation_size,
				batch_size)
		]

		def Kfold_input_fn_creater(foldid):
			fold_examples = valid_splits[foldid]
			valid_fold_data = [
				valid_data[i] for i in range(len(valid_data)) if i in fold_examples
			]

			def Kfold_input_fn(params):
				map_to_image_label_given_pixel = functools.partial(map_to_image_label,
					pixel=params['pixel'])
				valid_dataset = tf.data.Dataset.from_tensor_slices(valid_fold_data)
				valid_dataset = valid_dataset.map(map_to_image_label_given_pixel)
				valid_dataset = valid_dataset.batch(len(valid_fold_data))
				return valid_dataset
			return Kfold_input_fn
	else:
		Kfold_input_fn_creater = None

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, asym=False):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		shifted_test_data = shifted_data_dict[py]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater
