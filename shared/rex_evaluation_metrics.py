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

"""Evaluation metrics for the main method."""
import copy

from shared import losses

import tensorflow as tf

def compute_pred_loss(labels, logits, sample_weights, params):

	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)
	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	if params['weighted_mmd'] == 'False':
		return tf.reduce_mean(individual_losses)

	weighted_loss = sample_weights * individual_losses
	weighted_loss = tf.math.divide_no_nan(
		tf.reduce_sum(weighted_loss),
		tf.reduce_sum(sample_weights)
	)
	return weighted_loss


def compute_loss(labels, logits, z_pred, sample_weights,
	sample_weights_pos, sample_weights_neg, params):
	del z_pred, sample_weights, sample_weights_pos, sample_weights_neg

	# --- get the loss for all the examples 
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)
	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	if len(individual_losses.shape) == 1:
		individual_losses = tf.expand_dims(individual_losses, axis=-1)

	# --- get the loss for each group 
	auxiliary_labels = tf.expand_dims(labels[:, 1], axis=-1)

	pos_pos_mask = y_main * auxiliary_labels
	pos_neg_mask = y_main * (1.0 - auxiliary_labels)
	neg_pos_mask = (1.0 - y_main) * auxiliary_labels
	neg_neg_mask = (1.0 - y_main) * (1.0 - auxiliary_labels)


	R_pos_pos = tf.math.divide_no_nan(
		tf.reduce_sum(individual_losses * pos_pos_mask),
		tf.reduce_sum(pos_pos_mask))
	R_pos_neg = tf.math.divide_no_nan(
		tf.reduce_sum(individual_losses * pos_neg_mask),
		tf.reduce_sum(pos_neg_mask))
	R_neg_pos = tf.math.divide_no_nan(
		tf.reduce_sum(individual_losses * neg_pos_mask),
		tf.reduce_sum(neg_pos_mask))
	R_neg_neg = tf.math.divide_no_nan(
		tf.reduce_sum(individual_losses * neg_neg_mask),
		tf.reduce_sum(neg_neg_mask))


	# --- compute sum 
	R_sum = R_pos_pos + R_pos_neg + R_neg_pos + R_neg_neg

	# --- compute variance 
	R_variance = tf.math.reduce_variance(
		[R_pos_pos, R_pos_neg, R_neg_pos, R_neg_neg])

	return R_sum, R_variance


def get_prediction_by_group(labels, predictions):

	mean_prediction_dict = {}

	labels11_mask = tf.where(labels[:, 0] * labels[:, 1])
	mean_prediction_dict['mean_pred_11'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels11_mask)
	)

	labels10_mask = tf.where(labels[:, 0] * (1.0 - labels[:, 1]))
	mean_prediction_dict['mean_pred_10'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels10_mask)
	)

	labels01_mask = tf.where((1.0 - labels[:, 0]) * labels[:, 1])
	mean_prediction_dict['mean_pred_01'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels01_mask)
	)

	labels00_mask = tf.where((1.0 - labels[:, 0]) * (1.0 - labels[:, 1]))
	mean_prediction_dict['mean_pred_00'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels00_mask)
	)

	return mean_prediction_dict

def get_accuracy_by_group(labels, predicted_class):

	if len(predicted_class.shape) == 1:
		predicted_class = tf.expand_dims(predicted_class, axis=-1)


	y_main = tf.expand_dims(labels[:, 0], axis=-1)

	mis_class = (y_main == predicted_class) * 1.0 

	auxiliary_labels = tf.expand_dims(labels[:, 1], axis=-1)
	pos_pos_mask = y_main * auxiliary_labels
	pos_neg_mask = y_main * (1.0 - auxiliary_labels)
	neg_pos_mask = (1.0 - y_main) * auxiliary_labels
	neg_neg_mask = (1.0 - y_main) * (1.0 - auxiliary_labels)

	
	group_accuracy = {}

	
	group_accuracy['acc_11'] = tf.compat.v1.metrics.mean(
		tf.math.divide_no_nan(
		tf.reduce_sum(y_main * pos_pos_mask), 
		tf.reduce_sum(pos_pos_mask)
	))

	group_accuracy['acc_10'] = tf.compat.v1.metrics.mean(
		tf.math.divide_no_nan(
		tf.reduce_sum(y_main * pos_neg_mask), 
		tf.reduce_sum(pos_neg_mask)
	))

	group_accuracy['acc_01'] = tf.compat.v1.metrics.mean(
		tf.math.divide_no_nan(
		tf.reduce_sum(y_main * neg_pos_mask), 
		tf.reduce_sum(neg_pos_mask)
	))

	group_accuracy['acc_00'] = tf.compat.v1.metrics.mean(
		tf.math.divide_no_nan(
		tf.reduce_sum(y_main * neg_neg_mask), 
		tf.reduce_sum(neg_neg_mask)
	))

	return group_accuracy




def auroc(labels, predictions):
	""" Computes AUROC """
	auc_metric = tf.keras.metrics.AUC(name="auroc")
	auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric

def get_eval_metrics_dict(labels, predictions, sample_weights,
	sample_weights_pos, sample_weights_neg, sigma_list, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	eval_metrics_dict = {}

	# -- the "usual" evaluation metrics
	eval_metrics_dict['accuracy'] = tf.compat.v1.metrics.accuracy(
		labels=y_main, predictions=predictions["classes"])

	eval_metrics_dict["auc"] = auroc(
		labels=y_main, predictions=predictions["probabilities"])

	# -- Mean predictions for each group
	mean_prediction_by_group = get_prediction_by_group(labels,
		predictions["probabilities"])

	# --- Accuracy for each group 
	accuracy_by_group = get_accuracy_by_group(labels, 
		predictions["classes"])

	return {**eval_metrics_dict, **mean_prediction_by_group, **accuracy_by_group}


