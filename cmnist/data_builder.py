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

"""Functions to create a corrupted MNIST dataset."""

import numpy as np
import tensorflow as tf


def color_corrupt_img(x, y, npix=5):
	"""Corrupts a single MNIST image.

	Adds RGB color channels to greyscale mnist images. Then corrupts the image by
	changing the color of randomly chosen bright pixels. If y=0, we add
	corruptions to channel 1. If y=1, we add corruptions to channel 2

	Args:
		x: image
		y: label = 1 or 0
		npix: number of pixels to corrupt

	Returns:
		corrupted image
	"""
	# we will corrupt channel 1 if y = 0 and channel 2 if y ==1
	channel_to_corrupt = int(y) + 1

	# add color channels
	xc = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)).numpy()
	bright_pixels = np.where(xc[:, :, channel_to_corrupt] > .5)

	# pick npix to corrupt
	if npix > len(bright_pixels[0]):
		pick_pix = list(range(len(bright_pixels[0])))
	else:
		pick_pix = np.random.choice(len(bright_pixels[0]), size=npix, replace=False)
	pr, pc = bright_pixels[0][pick_pix], bright_pixels[1][pick_pix]
	xc[pr[:, None], pc, channel_to_corrupt] = 0
	return xc


def corrupt_mnist(x, y, py1_y0, pflip0=.1, pflip1=.1, npix=5, rng=None):
	"""Creates a corrupted mnist dataset (images and labels).

	We only use images with digit labels = 3, 4, 5, 6 from the original mnist
	data. For images with digit labels 3,4 we corrupt pixels on channel 1. For 5,6
	we corrupt pixels on channel 2. For each image 2 labels are generated: y0 and
	y1. With probability 1-pflip: y0 = 1 if digit = 3 or 4 and 0 otherwise. With
	probability py1_y0: y1 is 1 if digit = 3 or 4, and 0 otherwise.

	Args:
		x: tensor of clean mnist images
		y: vector of original mnist labels
		py1_y0: scalar, probabiltiy of y_1 = 1| y_0 = 1
		pflip0: scalar, probability of flipping y0 label
		pflip1: scalar, probability of flipping y1 label
		npix: number of pixels to corrupt
		rng: numpy.random.RandomState

	Returns:
		corrupted images, and matrix of outcomes.
	"""
	if rng is None:
		rng = np.random.RandomState(0)

	keep = (y == 3) | (y == 4)
	x, y = x[keep].copy(), y[keep].copy()
	y0_true = ((y == 3)) * 1

	y0 = y0_true.copy()
	if pflip0 > 0:
		flips = rng.choice(
			range(y0.shape[0]), size=int(pflip0 * y0.shape[0]), replace=False)
		y0[flips] = 1 - y0[flips]

	if py1_y0 in [0, 1]:
		y1 = y0_true * py1_y0 + (1 - y0_true) * (1 - py1_y0)
	else:
		y1 = rng.binomial(
			1, y0_true * py1_y0 + (1 - y0_true) * (1 - py1_y0), size=y0_true.shape)
	if pflip1 > 0:
		flips = rng.choice(
			range(y0.shape[0]), size=int(pflip1 * y0.shape[0]), replace=False)
		y0[flips] = 1 - y0[flips]
	xc = [color_corrupt_img(img, y=lab, npix=npix) for img, lab in zip(x, y1)]
	xc = np.stack(xc, axis=0)
	xc = xc.astype(np.float32)

	y_mat = np.stack([y0, y1], axis=1)
	y_mat = y_mat.astype(np.float32)

	return xc, y_mat


def get_corrupt_minst(p_tr=.7,
											py1_y0=1,
											py1_y0_s=.5,
											pflip0=.1,
											pflip1=.1,
											npix=5,
											oracle_prop=0.0,
											random_seed=None):
	"""Gets train, vald, test data.

	Args:
		p_tr: float, proportion of train data used for training (vs vald)
		py1_y0: float, probability of y1=1 | y0 for the main dist
		py1_y0_s: float, probability of y1=1 | y0 for shifted dist
		pflip0: float, probability of flipping y0 label
		pflip1: float, probability of flipping y1 label
		oracle_prop: float, proportion (relative to training size) to add
			as oracle augmentation
		npix: number of pixels to corrupt
		random_seed: seed for numpy.random.RandomState

	Returns:
		train, vald, test, shifted_test data
	"""
	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	(x_train_valid_or,
		y_train_valid_or), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train_valid_or, x_test = x_train_valid_or[..., np.newaxis] / 255.0, x_test[
		..., np.newaxis] / 255.0

	# get training and validation data sampled according to observation
	# distribution
	x_train_valid, y_train_valid = corrupt_mnist(
		x_train_valid_or,
		y_train_valid_or,
		py1_y0=py1_y0,
		pflip0=pflip0,
		pflip1=pflip1,
		npix=npix,
		rng=rng)

	if oracle_prop > 0:
		x_train_valid_aug, y_train_valid_aug = corrupt_mnist(
			x_train_valid_or,
			y_train_valid_or,
			py1_y0=py1_y0_s,
			pflip0=pflip0,
			pflip1=pflip1,
			npix=npix,
			rng=rng)

		augmentation_samples = np.random.choice(x_train_valid.shape[0],
			size=int(oracle_prop * x_train_valid.shape[0]), replace=False).tolist()

		x_train_valid_aug = x_train_valid_aug[augmentation_samples, :, :]
		y_train_valid_aug = y_train_valid_aug[augmentation_samples]

		x_train_valid = np.concatenate([x_train_valid, x_train_valid_aug], axis=0)
		y_train_valid = np.concatenate([y_train_valid, y_train_valid_aug], axis=0)

		shuffle_ids = np.random.choice(x_train_valid.shape[0],
			size=x_train_valid.shape[0], replace=False).tolist()
		x_train_valid = x_train_valid[shuffle_ids, :, :]
		y_train_valid = y_train_valid[shuffle_ids, :]

	# split into separate training and testing
	train_ind = rng.choice(
		x_train_valid.shape[0],
		size=int(p_tr * x_train_valid.shape[0]),
		replace=False)

	valid_ind = list(set(range(x_train_valid.shape[0])) - set(train_ind))
	train_data = x_train_valid[train_ind], y_train_valid[train_ind]
	valid_data = x_train_valid[valid_ind], y_train_valid[valid_ind]

	# get testing data sampled according to same distribution as observation
	# distribution
	same_test_data = corrupt_mnist(
		x_test,
		y_test,
		py1_y0=py1_y0,
		pflip0=pflip0,
		pflip1=pflip1,
		npix=npix,
		rng=rng)

	# get testing data sampled according to a shifted distribution i.e.,
	# different from observation distribution
	shifted_test_data = corrupt_mnist(
		x_test,
		y_test,
		py1_y0=py1_y0_s,
		pflip0=pflip0,
		pflip1=pflip1,
		npix=npix,
		rng=rng)

	return train_data, valid_data, same_test_data, shifted_test_data


def build_input_fns(p_tr=.7,
										py1_y0=1,
										py1_y0_s=.5,
										pflip0=.1,
										pflip1=.1,
										npix=5,
										oracle_prop=0.0,
										random_seed=None):
	"""Builds datasets for train, eval, and test.

	Args:
		p_tr: scalar, proportion of train data used for training (vs vald)
		py1_y0: scalar, probability of y1=1 | y0 for the main dist
		py1_y0_s: scalar, probability of y1=1 | y0 for shifted dist
		pflip0: scalar, probability of flipping y0 label
		pflip1: scalar, probability of flipping y1 label
		npix: number of pixels to corrupt
		oracle_prop: float, proportion (relative to training size) to add
			as oracle augmentation
		random_seed: seed for numpy.random.RandomState

	Returns:
		train, vald, test, shifted_test data iterators
	"""
	# TODO(mmakar) add unit test for this function
	all_data = get_corrupt_minst(
		p_tr=p_tr,
		py1_y0=py1_y0,
		py1_y0_s=py1_y0_s,
		pflip0=pflip0,
		pflip1=pflip1,
		npix=npix,
		oracle_prop=oracle_prop,
		random_seed=random_seed)
	train_data, valid_data, same_test_data, shifted_test_data = all_data

	# Build an iterator over training batches.
	def train_input_fn(params):
		batch_size = params['batch_size']
		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.shuffle(int(1e5)).batch(batch_size).repeat(
			params['num_epochs'])
		return dataset

	# Build an iterator over validation batches
	def valid_input_fn(params):
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.batch(batch_size).repeat(1)
		return valid_dataset

	# Build an iterator over the heldout set (same distribution).
	def eval_input_fn():
		batch_size = int(1e5)
		eval_dataset = tf.data.Dataset.from_tensor_slices(same_test_data)
		eval_dataset = eval_dataset.batch(batch_size).repeat(1)
		return eval_dataset

	# Build an iterator over the heldout set (shifted distribution).
	def eval_shift_input_fn():
		batch_size = int(1e5)
		eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
		eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
		return eval_shift_dataset

	return train_input_fn, valid_input_fn, eval_input_fn, eval_shift_input_fn
