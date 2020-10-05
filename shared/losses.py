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

"""Library of non-typical losses."""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

def mmd_loss(embedding, main_labels, auxiliary_labels, sigma, weighted):
	if weighted:
		return mmd_loss_weighted(embedding, main_labels, auxiliary_labels, sigma)
	return mmd_loss_unweighted(embedding, auxiliary_labels, sigma)


def mmd_loss_unweighted(embedding, auxiliary_labels, sigma):
	r"""Computes MMD loss between embeddings of groups defined by label.

	Maximum Mean Discrepancy (MMD) is an integrated probability metric.
	It measures the distance between two probability distributions. In this
	setting, we measure (and penalize) the distance between the probability
	distributions of the embeddings of group 0 (where auxiliary_labels ==0), and
	the emeddings of group 1 (where auxiliary_labels ==1). The specific equation
	for computing the MMD is:

	MMD^2(P, Q)= || \E{\phi_sigma(x)} - \E{\phi_sigma(y)} ||^2
						= \E{ K_sigma(x, x) } + \E{ K_sigma(y, y) } - 2 \E{ K_sigma(x, y)},

	where K_sigma = <\phi_sigma(x), \phi_sigma(y)>,is a kernel function,
	in this case a radial basis kernel, with bandwidth sigma.

	For our main approach, we penalize the mmd_loss (encourage the distance to be
	small i.e., encourage the two distributions to be close, which is roughly
	equivalent to an adversarial setting: by forcing MMD to be small, an
	adversary cannot distinguish between the two groups solely based on the
	embedding. This also implies that cross-prediction (predicting
	auxiliary_labels using embedding) is penalized.

	Args:
		embedding: tensor with learned embedding
		auxiliary_labels: tensor with label defining 2 groups
		sigma: scalar, bandwidth for kernel used to compute MMD
	Returns:
		MMD between embeddings of the two groups defined by label
	"""
	kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
		amplitude=1.0, length_scale=sigma)

	kernel_mat = kernel.matrix(embedding, embedding)

	if len(auxiliary_labels.shape) == 1:
		auxiliary_labels = tf.expand_dims(auxiliary_labels, axis=-1)

	pos_mask = tf.matmul(auxiliary_labels, tf.transpose(auxiliary_labels))
	neg_mask = tf.matmul(1.0 - auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	pos_neg_mask = tf.matmul(auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))

	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_mask * kernel_mat), tf.reduce_sum(pos_mask))
	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_mask * kernel_mat), tf.reduce_sum(neg_mask))
	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_mask * kernel_mat), tf.reduce_sum(pos_neg_mask))

	mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
	mmd_val = tf.maximum(0.0, mmd_val)

	return mmd_val


def mmd_loss_weighted(embedding, main_labels, auxiliary_labels, sigma):
	r"""Computes MMD loss between embeddings of groups defined by label.

	Maximum Mean Discrepancy (MMD) is an integrated probability metric.
	It measures the distance between two probability distributions. In this
	setting, we measure (and penalize) the distance between the probability
	distributions of the embeddings of group 0 (where auxiliary_labels ==0), and
	the emeddings of group 1 (where auxiliary_labels ==1). The specific equation
	for computing the MMD is:

	MMD^2(P, Q)= || \E{\phi_sigma(x)} - \E{\phi_sigma(y)} ||^2
						= \E{ K_sigma(x, x) } + \E{ K_sigma(y, y) } - 2 \E{ K_sigma(x, y)},

	where K_sigma = <\phi_sigma(x), \phi_sigma(y)>,is a kernel function,
	in this case a radial basis kernel, with bandwidth sigma.

	For our main approach, we penalize the mmd_loss (encourage the distance to be
	small i.e., encourage the two distributions to be close, which is roughly
	equivalent to an adversarial setting: by forcing MMD to be small, an
	adversary cannot distinguish between the two groups solely based on the
	embedding. This also implies that cross-prediction (predicting
	auxiliary_labels using embedding) is penalized

	Args:
		embedding: tensor with learned embedding
		main_labels: tensor with main labels
		auxiliary_labels: tensor with label defining 2 groups
		sigma: scalar, bandwidth for kernel used to compute MMD
	Returns:
		MMD between embeddings of the two groups defined by label
	"""

	kernel_mat = tfa.losses.metric_learning.pairwise_distance(embedding,
		squared=True)
	kernel_mat = tf.math.exp((- kernel_mat) / sigma)

	weights_pos = tf.multiply(auxiliary_labels, main_labels)
	# TODO : what happens with nans?
	weights_pos = tf.math.divide_no_nan(tf.reduce_sum(weights_pos),
		tf.reduce_sum(auxiliary_labels))
	weights_pos = tf.clip_by_value(weights_pos, 1e-5, 1 - 1e-5)
	weights_pos = main_labels * weights_pos + (
		1.0 - main_labels) * (1.0 - weights_pos)
	weights_pos = 1.0 / weights_pos
	weights_pos = weights_pos * auxiliary_labels

	weights_neg = tf.multiply((1.0 - auxiliary_labels), main_labels)
	# TODO : what happens with nans?
	weights_neg = tf.math.divide_no_nan(tf.reduce_sum(weights_neg),
		tf.reduce_sum((1.0 - auxiliary_labels)))
	weights_neg = tf.clip_by_value(weights_neg, 1e-5, 1 - 1e-5)
	weights_neg = main_labels * weights_neg + (
		1.0 - main_labels) * (1.0 - weights_neg)
	weights_neg = 1.0 / weights_neg
	weights_neg = weights_neg * (1.0 - auxiliary_labels)

	if len(auxiliary_labels.shape) == 1:
		auxiliary_labels = tf.expand_dims(auxiliary_labels, axis=-1)

	pos_mask = tf.matmul(auxiliary_labels, tf.transpose(auxiliary_labels))
	neg_mask = tf.matmul(1.0 - auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	pos_neg_mask = tf.matmul(auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	neg_pos_mask = tf.matmul((1.0 - auxiliary_labels),
		tf.transpose(auxiliary_labels))

	pos_kernel_mean = kernel_mat * pos_mask
	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_kernel_mean, axis=1),
		tf.reduce_sum(pos_mask, axis=1))
	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_kernel_mean * weights_pos),
		tf.reduce_sum(weights_pos) * sigma)

	neg_kernel_mean = kernel_mat * neg_mask
	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_kernel_mean, axis=1),
		tf.reduce_sum(neg_mask, axis=1))
	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_kernel_mean * weights_neg),
		tf.reduce_sum(weights_neg) * sigma)

	neg_pos_kernel_mean = kernel_mat * neg_pos_mask
	neg_pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_pos_kernel_mean, axis=1),
		tf.reduce_sum(neg_pos_mask, axis=1))
	neg_pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_pos_kernel_mean * weights_neg),
		tf.reduce_sum(weights_neg) * sigma)

	pos_neg_kernel_mean = kernel_mat * pos_neg_mask
	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_kernel_mean, axis=1),
		tf.reduce_sum(pos_neg_mask, axis=1))
	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_kernel_mean * weights_pos),
		tf.reduce_sum(weights_pos) * sigma)

	mmd_val = pos_kernel_mean + neg_kernel_mean - (
		pos_neg_kernel_mean + neg_pos_kernel_mean)
	mmd_val = tf.maximum(0.0, mmd_val)

	return mmd_val


