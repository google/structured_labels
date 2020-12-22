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

"""Main training protocol used for structured label prediction models."""
import glob
import os
import pickle
import copy

from shared import architectures
from shared import losses

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class EvalCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
	""" Allows evaluation on multiple datasets """
	def __init__(self, estimator, input_fn, name):
		self.estimator = estimator
		self.input_fn = input_fn
		self.name = name

	def after_save(self, session, global_step):
		print("RUNNING EVAL: {}".format(self.name))
		self.estimator.evaluate(self.input_fn, name=self.name)
		print("FINISHED EVAL: {}".format(self.name))


def auroc(labels, predictions):
	""" Computes AUROC """
	auc_metric = tf.keras.metrics.AUC(name="auroc")
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric

# def compute_mean_predictions(labels, predictions):

def flatten_dict(dd, separator='_', prefix=''):
	""" Flattens the dictionary with eval metrics """
	return {
		prefix + separator + k if prefix else k: v
		for kk, vv in dd.items()
		for k, v in flatten_dict(vv, separator, kk).items()
	} if isinstance(dd,
		dict) else {prefix: dd}


def cleanup_directory(directory):
	"""Deletes all files within directory and its subdirectories.

	Args:
		directory: string, the directory to clean up
	Returns:
		None
	"""
	if os.path.exists(directory):
		files = glob.glob(f"{directory}/*", recursive=True)
		for f in files:
			if os.path.isdir(f):
				os.system(f"rm {f}/* ")
			else:
				os.system(f"rm {f}")


def compute_loss(labels, logits, z_pred, sample_weights,
	sample_weights_pos, sample_weights_neg, params):
	if sample_weights is None:
		return compute_loss_unweighted(labels, logits, z_pred, params)
	return compute_loss_weighted(labels, logits, z_pred,
		sample_weights, sample_weights_pos, sample_weights_neg,  params)


def compute_loss_weighted(labels, logits, z_pred, sample_weights,
	sample_weights_pos, sample_weights_neg, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	# --- Prediction loss
	weighted_loss = sample_weights * individual_losses
	weighted_loss = tf.math.divide_no_nan(
		tf.reduce_sum(weighted_loss),
		tf.reduce_sum(sample_weights)
	)

	# --- MMD loss
	other_label_inds = [
		lab_ind for lab_ind in range(labels.shape[1])
		if lab_ind != params["label_ind"]
	]

	weighted_mmd_vals = []
	for lab_ind in other_label_inds:
		mmd_val = losses.mmd_loss(
			embedding=z_pred,
			auxiliary_labels=labels[:, lab_ind],
			weights_pos=sample_weights_pos,
			weights_neg=sample_weights_neg,
			params=params)
		weighted_mmd_vals.append(mmd_val[0])

	weighted_mmd = tf.concat(weighted_mmd_vals, axis=0)

	return weighted_loss, weighted_mmd


def compute_loss_unweighted(labels, logits, z_pred, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	# --- Prediction loss
	unweighted_loss = tf.reduce_mean(individual_losses)

	# --- MMD loss
	other_label_inds = [
		lab_ind for lab_ind in range(labels.shape[1])
		if lab_ind != params["label_ind"]
	]

	unweighted_mmd_vals = []
	for lab_ind in other_label_inds:
		mmd_val = losses.mmd_loss(
			embedding=z_pred,
			auxiliary_labels=labels[:, lab_ind],
			weights_pos=None,
			weights_neg=None,
			params=params)
		unweighted_mmd_vals.append(mmd_val[0])

	unweighted_mmd = tf.concat(unweighted_mmd_vals, axis=0)

	return unweighted_loss, unweighted_mmd


def serving_input_fn():
	feat = array_ops.placeholder(dtype=dtypes.float32)
	return tf.estimator.export.TensorServingInputReceiver(features=feat,
		receiver_tensors=feat)

def model_fn(features, labels, mode, params):
	""" Main training function ."""

	if params['architecture'] == 'simple':
		net = architectures.SimpleConvolutionNet(
			dropout_rate=params["dropout_rate"],
			l2_penalty=params["l2_penalty"],
			embedding_dim=params["embedding_dim"])
	elif params['architecture'] == 'pretrained_resnet':
		net = architectures.PretrainedResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])
	elif params['architecture'] == 'pretrained_resnet_random':
		net = architectures.RandomResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])

	elif params['architecture'] == 'pretrained_resnet101':
		net = architectures.PretrainedResNet101(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])

	elif params['architecture'] == 'from_scratch_resnet':
		net = architectures.ScratchResNet50()

	training_state = mode == tf.estimator.ModeKeys.TRAIN
	logits, zpred = net(features, training=training_state)
	ypred = tf.nn.sigmoid(logits)

	predictions = {
		"classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
		"logits": logits,
		"probabilities": ypred,
		"embedding": zpred
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				"classify": tf.estimator.export.PredictOutput(predictions)
			})

	if labels['labels'].shape[1] != 2:
		raise NotImplementedError('Only 2 labels supported for now')

	if (params['weighted_mmd'] == 'True') and (params['balanced_weights'] == 'True'):
		sample_weights_pos = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 0]), axis=-1)
		sample_weights_neg = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 1]), axis=-1)
		sample_weights = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 2]), axis=-1)
	else:
		sample_weights_pos = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 0]), axis=-1)
		sample_weights_neg = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 1]), axis=-1)
		sample_weights = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 2]), axis=-1)

	labels = tf.identity(labels['labels'])

	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	if mode == tf.estimator.ModeKeys.EVAL:
		if params['minimize_logits'] == 'True':
			unweighted_loss, unweighted_mmd = compute_loss(labels, logits, logits, None,
				None, None, params)
			weighted_loss, weighted_mmd = compute_loss(labels, logits, logits,
				sample_weights, sample_weights_pos, sample_weights_neg, params)

		else:
			unweighted_loss, unweighted_mmd = compute_loss(labels, logits, zpred, None,
				None, None, params)
			weighted_loss, weighted_mmd = compute_loss(labels, logits, zpred,
				sample_weights, sample_weights_pos, sample_weights_neg, params)
		# -- compute actual loss
		if params['weighted_mmd'] == "True":
			loss = weighted_loss + params["alpha"] * weighted_mmd
		else:
			loss = unweighted_loss + params["alpha"] * unweighted_mmd

		extra_metric_evals = {}
		orig_sigma = params['sigma']
		orig_weighting = params['weighted_mmd']
		for sigma_val in [0.1, 1, 10, 100, 1000, 10000]:
			uw_temp_params = copy.deepcopy(params)
			uw_temp_params['sigma'] = sigma_val
			uw_temp_params['weighted_mmd'] = 'False'
			_, uw_mmd_val_at_sigma = compute_loss(labels, logits, zpred, None,
				None, None, uw_temp_params)
			extra_metric_evals[f'uw_mmd{sigma_val}'] = tf.compat.v1.metrics.mean(
				uw_mmd_val_at_sigma)

			w_temp_params = copy.deepcopy(params)
			w_temp_params['sigma'] = sigma_val
			w_temp_params['weighted_mmd'] = 'True'
			_, w_mmd_val_at_sigma = compute_loss(labels, logits, zpred,
				sample_weights, sample_weights_pos, sample_weights_neg,
				w_temp_params)
			extra_metric_evals[f'w_mmd{sigma_val}'] = tf.compat.v1.metrics.mean(
				w_mmd_val_at_sigma)

		assert params['sigma'] == orig_sigma
		assert params['weighted_mmd'] == orig_weighting

		main_1_mask = tf.where(labels[:, 0])
		main_1_zpred = tf.gather(zpred, main_1_mask)
		main_1_auxiliary_labels = tf.gather(labels[:, 1], main_1_mask)
		sample_weights_pos_1 = tf.gather(sample_weights_pos, main_1_mask)
		sample_weights_neg_1 = tf.gather(sample_weights_neg, main_1_mask)

		uw_mmd_class1 = losses.mmd_loss(
			embedding=main_1_zpred,
			auxiliary_labels=main_1_auxiliary_labels,
			weights_pos=None,
			weights_neg=None,
			params=params)

		w_mmd_class_1 = losses.mmd_loss(
			embedding=main_1_zpred,
			auxiliary_labels=main_1_auxiliary_labels,
			weights_pos=sample_weights_pos_1,
			weights_neg=sample_weights_neg_1,
			params=params)

		main_0_mask = tf.where(1.0 - labels[:, 0])
		main_0_zpred = tf.gather(zpred, main_0_mask)
		main_0_auxiliary_labels = tf.gather(labels[:, 1], main_0_mask)
		sample_weights_pos_0 = tf.gather(sample_weights_pos, main_0_mask)
		sample_weights_neg_0 = tf.gather(sample_weights_neg, main_0_mask)

		uw_mmd_class_0 = losses.mmd_loss(
			embedding=main_0_zpred,
			auxiliary_labels=main_0_auxiliary_labels,
			weights_pos=None,
			weights_neg=None,
			params=params)

		w_mmd_class_0 = losses.mmd_loss(
			embedding=main_0_zpred,
			auxiliary_labels=main_0_auxiliary_labels,
			weights_pos=sample_weights_pos_0,
			weights_neg=sample_weights_neg_0,
			params=params)

		labels11_mask = tf.where(labels[:, 0] * labels[:, 1])
		mean_pred_11 = tf.gather(ypred, labels11_mask)
		labels10_mask = tf.where(labels[:, 0] * (1.0 - labels[:, 1]))
		mean_pred_10 = tf.gather(ypred, labels10_mask)
		labels01_mask = tf.where((1.0 - labels[:, 0]) * labels[:, 1])
		mean_pred_01 = tf.gather(ypred, labels01_mask)
		labels00_mask = tf.where((1.0 - labels[:, 0]) * (1.0 - labels[:, 1]))
		mean_pred_00 = tf.gather(ypred, labels00_mask)

		extra_metric_evals['uw_mmd_class1'] = tf.compat.v1.metrics.mean(
			uw_mmd_class1[0])
		extra_metric_evals['w_mmd_class_1'] = tf.compat.v1.metrics.mean(
			w_mmd_class_1[0])
		extra_metric_evals['uw_mmd_class_0'] = tf.compat.v1.metrics.mean(
			uw_mmd_class_0[0])
		extra_metric_evals['w_mmd_class_0'] = tf.compat.v1.metrics.mean(
			w_mmd_class_0[0])
		extra_metric_evals['mean_pred_11'] = tf.compat.v1.metrics.mean(
			mean_pred_11)
		extra_metric_evals['mean_pred_10'] = tf.compat.v1.metrics.mean(
			mean_pred_10)
		extra_metric_evals['mean_pred_01'] = tf.compat.v1.metrics.mean(
			mean_pred_01)
		extra_metric_evals['mean_pred_00'] = tf.compat.v1.metrics.mean(
			mean_pred_00)

		eval_metrics = {
			"accuracy":
				tf.compat.v1.metrics.accuracy(
					labels=y_main, predictions=predictions["classes"]),
			"auc": auroc(labels=y_main, predictions=predictions["probabilities"]),

			"unweighted_loss": tf.compat.v1.metrics.mean(unweighted_loss),
			"weighted_loss": tf.compat.v1.metrics.mean(weighted_loss),

			"unweighted_mmd": tf.compat.v1.metrics.mean(unweighted_mmd),
			"weighted_mmd": tf.compat.v1.metrics.mean(weighted_mmd)
		}

		eval_metrics = {**eval_metrics, **extra_metric_evals}

		# logging_hook = tf.estimator.LoggingTensorHook(every_n_iter=1,
		# 	tensors={
		# 		'labs': tf.reduce_max(labels.shape),
		# 		'mean_class': tf.reduce_max(sample_weights.shape)
		# 	})

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=None, eval_metric_ops=eval_metrics)

	if mode == tf.estimator.ModeKeys.TRAIN:

		opt = tf.keras.optimizers.Adam()
		global_step = tf.compat.v1.train.get_global_step()

		ckpt = tf.train.Checkpoint(
			step=global_step, optimizer=opt, net=net)

		sgima_decayed = tf.convert_to_tensor(params['sigma']) * tf.cast(
			global_step, dtype=tf.float32)

		tf.compat.v1.summary.scalar('sgima_decayed',
				sgima_decayed)

		with tf.GradientTape() as tape:
			logits, zpred = net(features, training=training_state)
			ypred = tf.nn.sigmoid(logits)

			if params['minimize_logits'] == 'True':
				if params['weighted_mmd'] == "True":
					prediction_loss, mmd_loss = compute_loss(labels, logits, logits,
						sample_weights, sample_weights_pos, sample_weights_neg, params)
				else:
					prediction_loss, mmd_loss = compute_loss(labels, logits, logits,
						None, None, None, params)

			else:
				if params['weighted_mmd'] == "True":
					prediction_loss, mmd_loss = compute_loss(labels, logits, zpred,
						sample_weights, sample_weights_pos, sample_weights_neg, params)
				else:
					prediction_loss, mmd_loss = compute_loss(labels, logits, zpred,
						None, None, None, params)

			regularization_loss = tf.reduce_sum(net.losses)
			loss = regularization_loss + prediction_loss + params["alpha"] * mmd_loss

			mmd_val_op = tf.compat.v1.summary.scalar('mmd', mmd_loss)
			pred_loss_op = tf.compat.v1.summary.scalar('ploss', prediction_loss)
			regularization_loss_op = tf.compat.v1.summary.scalar('regloss',
				regularization_loss)

		summary_hook_list = [sgima_decayed,
				mmd_val_op, pred_loss_op, regularization_loss_op]

		orig_sigma = params['sigma']
		orig_weighting = params['weighted_mmd']
		for sigma_val in [0.1, 1, 10, 100, 1000, 10000]:
			uw_temp_params = copy.deepcopy(params)
			uw_temp_params['sigma'] = sigma_val
			uw_temp_params['weighted_mmd'] = 'False'
			_, uw_mmd_val_at_sigma = compute_loss(labels, logits, zpred, None,
				None, None, uw_temp_params)

			summary_hook_list.append(
				tf.compat.v1.summary.scalar(f'uw_mmd{sigma_val}', uw_mmd_val_at_sigma)
			)

			w_temp_params = copy.deepcopy(params)
			w_temp_params['sigma'] = sigma_val
			w_temp_params['weighted_mmd'] = 'True'
			_, w_mmd_val_at_sigma = compute_loss(labels, logits, zpred,
				sample_weights, sample_weights_pos, sample_weights_neg,
				w_temp_params)

			summary_hook_list.append(
				tf.compat.v1.summary.scalar(f'w_mmd{sigma_val}', w_mmd_val_at_sigma)
			)

		# assert params['sigma'] == orig_sigma
		# assert params['weighted_mmd'] == orig_weighting

		variables = net.trainable_variables
		gradients = tape.gradient(loss, variables)

		summary_hook = tf.estimator.SummarySaverHook(save_steps=100,
			summary_op=summary_hook_list)

		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=tf.group(
				opt.apply_gradients(zip(gradients, variables)),
				ckpt.step.assign_add(1)),
			training_hooks=[summary_hook])

def train(exp_dir,
					dataset_builder,
					architecture,
					training_steps,
					pixel,
					batch_size,
					alpha,
					sigma,
					weighted_mmd,
					balanced_weights,
					dropout_rate,
					l2_penalty,
					embedding_dim,
					random_seed,
					minimize_logits,
					cleanup,
					py1_y0_shift_list=None):
	"""Trains the estimator."""

	scratch_exp_dir = exp_dir.replace('/data/ddmg/slabs/',
		'/data/scratch/mmakar/')
	if not os.path.exists(scratch_exp_dir):
		os.mkdir(scratch_exp_dir)

	cleanup_directory(scratch_exp_dir)

	params = {
		"pixel": pixel,
		"training_steps": training_steps,
		"architecture": architecture,
		"batch_size": batch_size,
		"alpha": alpha,
		"sigma": sigma,
		"weighted_mmd": weighted_mmd,
		"balanced_weights": balanced_weights,
		"dropout_rate": dropout_rate,
		"l2_penalty": l2_penalty,
		"embedding_dim": embedding_dim,
		"minimize_logits": minimize_logits,
		"label_ind": 0
	}

	run_config = tf.estimator.RunConfig(
		tf_random_seed=random_seed,
		save_checkpoints_steps=100,
		# save_checkpoints_secs=500,
		keep_checkpoint_max=2)

	est = tf.estimator.Estimator(
		model_fn, model_dir=scratch_exp_dir, params=params, config=run_config)

	input_fns = dataset_builder()
	train_input_fn, valid_input_fn, eval_input_fn_creater = input_fns

	est.train(train_input_fn, steps=training_steps,
		saving_listeners=[
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.1, params), "0.1"),
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.5, params), "0.5"),
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.95, params),
				"0.95"),
		]
	)
	validation_results = est.evaluate(valid_input_fn)
	if params["weighted_mmd"] == "True":
		validation_results["xv_loss"] = validation_results["weighted_loss"]
		validation_results["xv_mmd"] = validation_results["weighted_mmd"]
	else:
		validation_results["xv_loss"] = validation_results["unweighted_loss"]
		validation_results["xv_mmd"] = validation_results["unweighted_mmd"]

	all_results = {"validation": validation_results}

	if py1_y0_shift_list is not None:
		for py in py1_y0_shift_list:
			eval_input_fn = eval_input_fn_creater(py, params)
			distribution_results = est.evaluate(eval_input_fn, steps=1e5)
			if params["weighted_mmd"] == "True":
				distribution_results['loss'] = distribution_results['unweighted_loss']
			all_results[f'shift_{py}'] = distribution_results

	# save results
	savefile = f"{exp_dir}/performance.pkl"
	all_results = flatten_dict(all_results)
	pickle.dump(all_results, open(savefile, "wb"))

	# save model
	print(f'{exp_dir}/saved_model')
	est.export_saved_model(f'{exp_dir}/saved_model', serving_input_fn)

	# if cleanup == 'True':
	# 	cleanup_directory(scratch_exp_dir)
