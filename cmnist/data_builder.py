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
	print(file_path)
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
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


def color_corrupt_img(index, images, labels, save_directory, npix=5):
	"""Corrupts a single MNIST image."""
	# we will corrupt channel 1 if y = 0 and channel 2 if y ==1
	x = images[index]
	y = labels[index, 1]
	channel_to_corrupt = int(y) + 1

	# add color channels
	xc = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)).numpy()
	bright_pixels = np.where(xc[:, :, channel_to_corrupt] > 127)

	# pick npix to corrupt
	if npix > len(bright_pixels[0]):
		pick_pix = list(range(len(bright_pixels[0])))
	else:
		pick_pix = np.random.choice(len(bright_pixels[0]), size=npix, replace=False)
	pr, pc = bright_pixels[0][pick_pix], bright_pixels[1][pick_pix]
	xc[pr[:, None], pc, channel_to_corrupt] = 0
	xc = tf.keras.preprocessing.image.array_to_img(xc, scale=False)
	tf.keras.preprocessing.image.save_img( 
		path=f'{save_directory}/image_{index}.jpg', 
		x=xc, data_format='channels_last', scale=False)

	data = pd.DataFrame({
		'img_filename': f'{save_directory}/image_{index}.jpg', 
		'y0': labels[index, 0], 'y1':y
	}, index = [0])
	return data


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


def create_images_labels(data_group='train', subset_ids=None, 
	py1_y0=1, pflip0=.1, pflip1=.1, npix=5, rng=None, experiment_directory=''):
	if rng is None:
		rng = np.random.RandomState(0)

	if data_group == 'test':
		_, (x, y) = tf.keras.datasets.mnist.load_data()
	else: 
		(x, y), _ = tf.keras.datasets.mnist.load_data()	


	print(f'==== length of full {data_group} x: {x.shape[0]}, y: {y.shape[0]}')
	x = x[..., np.newaxis]
	keep = (y == 3) | (y == 4)
	x, y = x[keep].copy(), y[keep].copy()

	print(f'==== length after dropping non 3/4 {data_group} x: {x.shape[0]}, y: {y.shape[0]}')

	if subset_ids is not None: 
		x = x[subset_ids]
		y = y[subset_ids]
	
	print(f'==== length of subset {data_group} x: {x.shape[0]}, y: {y.shape[0]}')

	# -- get noisy main label 
	y0_true = (y == 3) * 1
	flip0 = rng.choice(x.shape[0], size=int(pflip0 * x.shape[0]), replace=False).tolist()
	y0 = y0_true.copy()
	y0[flip0] = 1 - y0[flip0]

	print(f'Y0: true mean {np.mean(y0_true)}, noisy {np.mean(y0)}')

	# -- get corruption type (noisy aux label)
	y1_true = rng.binomial(1, y0_true * py1_y0 + (1 - y0_true) * (1.0 - py1_y0))
	flip1 = rng.choice(x.shape[0],size=int(pflip1 * x.shape[0]), replace=False).tolist()
	y1 = y1_true.copy()
	y1[flip1] = 1 - y1[flip1]

	print(f'Y1: true mean {np.mean(y1_true)}, noisy {np.mean(y1)}')

	# --- loop through each image to corrupt 
	labels = np.stack([y0, y1], axis=1)

	save_directory = f'{experiment_directory}/{data_group}_images/'
	if not os.path.exists(save_directory):
			os.mkdir(save_directory)
	print(f'Got npix {npix}')
	color_corrupt_img_wrapper = functools.partial(color_corrupt_img, images=x, labels=labels,
		save_directory=save_directory, npix=npix)	

	data = []
	pool = multiprocessing.Pool(NUM_WORKERS)
	for row in tqdm.tqdm(pool.imap_unordered(color_corrupt_img_wrapper, range(x.shape[0])),
			total=x.shape[0] ):
			data.append(row)

	data = pd.concat(data, axis=0, ignore_index=True)

	print(f"===={data_group} xtab between the two labels=====")
	print(pd.crosstab(data['y0'], data['y1'])/data.shape[0])

	data = get_weights(data) 
	print(f"===={data_group} values of balanced weights=====")
	print(data['balanced_weights'].value_counts())

	return data


def save_created_data(data_frame, experiment_directory, filename):
	txt_df = data_frame.img_filename + \
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


def create_save_mnist_lists(experiment_directory, py0=0.5, p_tr=.7, py1_y0=1,
	py1_y0_s=.5, pflip0=.1, pflip1=.1, npix=5, random_seed=None):

	if py0 != 0.5:
		raise NotImplementedError("Only accepting values of 0.8 and 0.5 for now")

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)


	# --- create train validation split
	# TODO: dont hard code data size
	train_valid_n = 11973
	
	train_ids = rng.choice(train_valid_n, size=int(p_tr * train_valid_n), replace=False).tolist()

	print(f'==== length of training data {len(train_ids)}========')
	train_df = create_images_labels(data_group='train', subset_ids=train_ids, 
		py1_y0=py1_y0, pflip0=pflip0, pflip1=pflip1, npix=npix, 
		rng=rng, experiment_directory=experiment_directory)

	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	validation_ids = list(set(range(train_valid_n)) - set(train_ids))
	print(f'==== length of training data {len(validation_ids)}========')
	validation_df = create_images_labels(data_group='validation', subset_ids=validation_ids, 
		py1_y0=py1_y0, pflip0=pflip0, pflip1=pflip1, npix=npix, 
		rng=rng, experiment_directory=experiment_directory)

	save_created_data(validation_df, experiment_directory=experiment_directory,
		filename='validation')

	# --- create + save test data

	for py1_y0_s_val in py1_y0_s:
		curr_test_df = create_images_labels(data_group=f'test{py1_y0_s_val}',
			subset_ids=None, py1_y0=py1_y0_s_val, pflip0=pflip0, pflip1=pflip1, npix=npix, 
			rng=rng, experiment_directory=experiment_directory)

		save_created_data(curr_test_df, experiment_directory=experiment_directory,
			filename=f'test_shift{py1_y0_s_val}')


def get_or_create_train_subsample(experiment_directory, n):
	if not os.path.exists(f'{experiment_directory}/train_{n}_sample.txt'):
		train_data, _, _ = load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=None)


def build_input_fns(p_tr=.7, py0=0.8, py1_y0=1, py1_y0_s=.5, pflip0=.1,
	pflip1=.1, npix=5, oracle_prop=0.0, Kfolds=0, random_seed=None, n=1e5):


	experiment_directory = (f'{DATA_DIR}/experiment_data/'
		f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}_npix{npix}')

	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/train.txt'):
		if not os.path.exists(experiment_directory):
			os.mkdir(experiment_directory)

		create_save_mnist_lists(
			experiment_directory=experiment_directory,
			py0=py0,
			p_tr=p_tr,
			py1_y0=py1_y0,
			py1_y0_s=py1_y0_s,
			pflip0=pflip0,
			pflip1=pflip1,
			npix=npix, 
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
		# dataset = dataset.shuffle(int(1e5)).batch(batch_size).repeat(num_epochs)
		dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label, num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
		return valid_dataset

	# -- Create kfold splits
	if Kfolds > 0:
		effective_validation_size = int(int(len(valid_data) / Kfolds) * Kfolds)
		batch_size = int(effective_validation_size / Kfolds)

		valid_splits = np.random.choice(len(valid_data), size=effective_validation_size,
			replace=False).tolist()

		valid_splits = [
			valid_splits[i:i + batch_size] for i in range(0, effective_validation_size, batch_size)
		]

		def Kfold_input_fn_creater(foldid):
			fold_examples = valid_splits[foldid]
			valid_fold_data = [valid_data[i] for i in range(len(valid_data)) if i in fold_examples]

			def Kfold_input_fn(params):
				valid_dataset = tf.data.Dataset.from_tensor_slices(valid_fold_data)
				valid_dataset = valid_dataset.map(map_to_image_label)
				valid_dataset = valid_dataset.batch(len(valid_fold_data))
				return valid_dataset
			return Kfold_input_fn
	else:
		Kfold_input_fn_creater = None


	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params):
		shifted_test_data = shifted_data_dict[py]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater
