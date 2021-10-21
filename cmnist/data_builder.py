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

"""Functions to create a corrupted MNIST dataset.

Code based on https://github.com/kohpangwei/group_DRO/blob/master/
	dataset_scripts/generate_waterbirds.py
"""
import os, shutil
import functools
import multiprocessing

from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm


DATA_DIR = '/data/ddmg/slabs/cmnist'
NUM_WORKERS = 20

def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img

def read_decode_png(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=1)
	return img


def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label

def map_to_image_label(x):

	weights_included = len(x) > 3

	digit_image = x[0]
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
	img = read_decode_jpg(digit_image)
	# resize, rescale  image
	img = tf.image.resize(img, (28, 28))
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

	y11_weight = np.sum(
		data_frame.y0 * data_frame.y1) / np.sum(data_frame.y1)
	y01_weight = np.sum(
		(1.0 - data_frame.y0) * data_frame.y1) / np.sum(data_frame.y1)

	y10_weight = np.sum(
		data_frame.y0 * (1.0 - data_frame.y1)) / np.sum((1.0 - data_frame.y1))
	y00_weight = np.sum(
		(1.0 - data_frame.y0) * (1.0 - data_frame.y1)) / np.sum(
		(1.0 - data_frame.y1))

	# -- positive weights
	data_frame['weights_pos'] = data_frame.y0 * y11_weight + \
		(1.0 - data_frame.y0) * y01_weight
	data_frame['weights_pos'] = 1.0 / data_frame['weights_pos']
	data_frame['weights_pos'] = data_frame.y1 * data_frame['weights_pos']

	assert data_frame.weights_pos.isin([np.nan, np.inf, -np.inf]).sum() == 0

	data_frame['balanced_weights_pos'] = np.mean(data_frame.y0) * \
		data_frame.y0 * data_frame.weights_pos + \
		np.mean(1.0 - data_frame.y0) * (1.0 - data_frame.y0) * data_frame.weights_pos

	# -- negative weights
	data_frame['weights_neg'] = data_frame.y0 * y10_weight + \
		(1.0 - data_frame.y0) * y00_weight
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


def create_images_labels(group, digit_data_frame, py1_y0=1, pflip0=.1,
	pflip1=.1, pixel=20, experiment_directory='', rng=None):
	if rng is None:
		rng = np.random.RandomState(0)

	# -- add noise to the main label
	flip0 = rng.choice(digit_data_frame.shape[0],
		size=int(pflip0 * digit_data_frame.shape[0]), replace=False).tolist()
	digit_data_frame.y0.loc[flip0] = 1 - digit_data_frame.y0.loc[flip0]

	# channel to corrupt is the true value for y1
	digit_data_frame['channel_to_corrupt'] = rng.binomial(1,
		digit_data_frame.y0 * py1_y0 + (1 - digit_data_frame.y0) * (1.0 - py1_y0))
	digit_data_frame['y1'] = digit_data_frame['channel_to_corrupt'].copy()

	# -- add noise
	flip1 = rng.choice(digit_data_frame.shape[0],
		size=int(pflip1 * digit_data_frame.shape[0]), replace=False).tolist()
	digit_data_frame.y1.loc[flip1] = 1 - digit_data_frame.y1.loc[flip1]

	# -- channel 0 is boring.
	digit_data_frame['channel_to_corrupt'] += 1

	if not os.path.exists(f'{experiment_directory}/{group}'):
		os.mkdir(f'{experiment_directory}/{group}')
	# --- create and save the corrupted image
	for i in range(digit_data_frame.shape[0]):
		digit_image = digit_data_frame.image_filename[i]
		channel_to_corrupt = digit_data_frame.channel_to_corrupt[i]
		img = read_decode_png(digit_image)

		# add color channels
		xc = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img)).numpy()
		# bright_pixels = np.where(xc[:, :, channel_to_corrupt] > 127)
		# # pick pixel to corrupt
		# if pixel > len(bright_pixels[0]):
		# 	pick_pix = list(range(len(bright_pixels[0])))
		# else:
		# 	pick_pix = np.random.choice(len(bright_pixels[0]), size=pixel, replace=False)
		# pr, pc = bright_pixels[0][pick_pix], bright_pixels[1][pick_pix]
		# xc[pr[:, None], pc, channel_to_corrupt] = 0
		if channel_to_corrupt == 2:
			xc[20:30, :10, channel_to_corrupt] = 255
		if channel_to_corrupt == 1:
			xc[:10, 20:30, channel_to_corrupt] = 255
		# imgc = tf.keras.preprocessing.image.array_to_img(xc, scale=False)
		tf.keras.preprocessing.image.save_img(
			path=f"{experiment_directory}/{group}/image{i}.png",
			x=xc, data_format='channels_last', scale=False)

	digit_data_frame['image_filename'] = [
		f"{experiment_directory}/{group}/image{i}.png" for i in range(digit_data_frame.shape[0])
	]
	digit_data_frame.drop('channel_to_corrupt', axis=1, inplace = True)
	return digit_data_frame


def save_created_data(data_frame, experiment_directory, filename):
	txt_df = data_frame.image_filename + \
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


def load_created_data(experiment_directory, py1_y0_s):

	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt').values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = pd.read_csv(
		f'{experiment_directory}/validation.txt').values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	if py1_y0_s is None:
		return train_data, validation_data, None
	test_data_dict = {}
	for py1_y0_s_val in py1_y0_s:
		test_data = pd.read_csv(
			f'{experiment_directory}/test_shift{py1_y0_s_val}.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		test_data_dict[py1_y0_s_val] = test_data

	return train_data, validation_data, test_data_dict


def create_save_cmnist_lists(experiment_directory, py0=0.5, p_tr=.7, py1_y0=1,
	py1_y0_s=.5, pflip0=.1, pflip1=.1, pixel=20, random_seed=None):

	if py0 != 0.5:
		raise NotImplementedError("Only accepting values of 0.5 for now")

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	# this is created in /data/ddmg/slabs/creates_corrupt_mnist.ipynb
	df = pd.read_csv(f'{DATA_DIR}/data/data_table.csv')
	df = df.sample(frac=1, random_state=random_seed)
	df.reset_index(inplace=True, drop=True)

	# --- create train+validation vs test split
	train_val_ids = rng.choice(df.shape[0],
		size=int(p_tr * df.shape[0]), replace=False).tolist()
	df['train_valid_ids'] = 0
	df.train_valid_ids.loc[train_val_ids] = 1

	# --- get the train and validation data
	train_valid_df = df[(df.train_valid_ids == 1)].reset_index(drop=True)

	train_valid_df = create_images_labels(
		group='train', digit_data_frame=train_valid_df, py1_y0=py1_y0, pflip0=pflip0,
		pflip1=pflip1, pixel=pixel, experiment_directory=experiment_directory,
		rng=rng)

	# --- create train validation split
	# TODO don't hard code p_tr
	train_ids = rng.choice(train_valid_df.shape[0],
		size=int(0.75 * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1

	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	train_df = get_weights(train_df)
	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	valid_df = get_weights(valid_df)
	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename='validation')

	# --- create + save test data
	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)

	for py1_y0_s_val in py1_y0_s:
		curr_test_df = test_df.copy()
		curr_test_df = create_images_labels(
			group='test', digit_data_frame=curr_test_df, py1_y0=py1_y0_s_val,
			pflip0=pflip0, pflip1=pflip1, pixel=pixel,
			experiment_directory=experiment_directory, rng=rng)

		curr_test_df = get_weights(curr_test_df)

		save_created_data(curr_test_df, experiment_directory=experiment_directory,
			filename=f'test_shift{py1_y0_s_val}')


def build_input_fns(p_tr=.7, py0=0.9, py1_y0=1, py1_y0_s=.5, pflip0=.1,
	pflip1=.1, pixel=20, oracle_prop=0.0, Kfolds=0, random_seed=None):

	experiment_directory = (f'{DATA_DIR}/experiment_data/'
		f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}')
	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/train.txt'):
		if not os.path.exists(experiment_directory):
			os.mkdir(experiment_directory)

		create_save_cmnist_lists(
			experiment_directory=experiment_directory,
			py0=py0,
			p_tr=p_tr,
			py1_y0=py1_y0,
			py1_y0_s=py1_y0_s,
			pflip0=pflip0,
			pflip1=pflip1,
			pixel=pixel,
			random_seed=random_seed)

	if oracle_prop > 0.0:
		raise NotImplementedError("not yet")

	# --load splits
	train_data, valid_data, shifted_data_dict = load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=py1_y0_s)

	# --this helps auto-set training steps at train time
	training_data_size = len(train_data)

	# Build an iterator over training batches.
	def train_input_fn(params):
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.map(map_to_image_label, num_parallel_calls=1)
		dataset = dataset.shuffle(int(1e5)).batch(batch_size).repeat(num_epochs)
		# dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label,
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
				valid_dataset = tf.data.Dataset.from_tensor_slices(valid_fold_data)
				valid_dataset = valid_dataset.map(map_to_image_label)
				valid_dataset = valid_dataset.batch(len(valid_fold_data))
				return valid_dataset
			return Kfold_input_fn
	else:
		Kfold_input_fn_creater = None

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, asym=False):
		del asym
		shifted_test_data = shifted_data_dict[py]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater
