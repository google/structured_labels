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
import os
import pickle
import copy
import gc

from shared import architectures
from shared import train_utils
from shared import evaluation_metrics
from shared import profiler

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import array_ops, variable_scope
from tensorflow.python.framework import ops, dtypes

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

SIGMA_LIST = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


class EvalCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
	""" Allows evaluation on multiple datasets """
	def __init__(self, estimator, input_fn, name):
		self.estimator = estimator
		self.input_fn = input_fn
		self.name = name

	def after_save(self, session, global_step):
		del session, global_step
		if self.name == "train":
			self.estimator.evaluate(self.input_fn, name=self.name, steps=1)
		else:
			self.estimator.evaluate(self.input_fn, name=self.name)

def get_or_create_sigma_variable(inital_sigma):
	graph = ops.get_default_graph()
	sigma_variable = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
		'sigma_tensor')

	# -- if sigma already exists, return it
	if len(sigma_variable) == 1:
		return sigma_variable[0]

	# -- if there are multiple sigmas, that's bad!
	elif len(sigma_variable) > 1:
		raise RuntimeError("There are multiple sigma variable in this graph!")

	# -- if no sigma exists, create it
	return variable_scope.get_variable(
		'sigma_tensor',
		initializer=inital_sigma,
		dtype=dtypes.float32,
		trainable=False,
		collections=['sigma_tensor', ops.GraphKeys.GLOBAL_VARIABLES],
		use_resource=True)


class DynamicSigmaHook(tf.estimator.SessionRunHook):
	# https://github.com/tensorflow/tensorflow/blob/89310df4bc576e
	# e951b0d86fe61035e990483a2b/tensorflow/python/training/basic_session_run_hooks.py#L325
	def __init__(self, estimator, input_fn_creater, params, epoch_size, sigma_value):
		self.estimator = estimator
		self.input_fn_creater = input_fn_creater
		self.params = params
		self.sigma = sigma_value
		self.epoch_size = epoch_size

	def begin(self):
		self._global_step_tensor = tf.compat.v1.train.get_global_step()
		self.variable = get_or_create_sigma_variable(self.sigma)

	def _update_sigma(self, value, session):
		return self.variable.load(value, session=session)

	def after_run(self, run_context, run_values):
		current_step_value = run_context.session.run(self._global_step_tensor)
		if current_step_value % self.epoch_size == 0:  # epoch
			results = []
			for foldid in range(self.params['Kfolds']):
				fold_results = self.estimator.evaluate(self.input_fn_creater(foldid),
					name=f'bandwidth_selection_{foldid}')
				if foldid == 0:
					print(f' ########## Changing sigma at global step {current_step_value} ###########')
					print(fold_results['global_step'])

				fold_results = {
					key: value for key, value in fold_results.items() if ('mmd' in key) and ('mmd' != key)
				}
				fold_results = pd.DataFrame(fold_results, index=[0])
				results.append(fold_results)

			results = pd.concat(results, axis=0)

			results_mean = results.mean(axis=0)
			results_std = results.std(axis=0)
			results_ratio = results_mean / results_std
			results_ratio[(results_std==0)] = 0

			print(results_ratio)
			results_ratio = results_ratio[(results_ratio == results_ratio.max())]
			print(results_ratio)
			best_sigma = results_ratio.index.tolist()
			best_sigma = [float(sigma_val.replace('mmd', '')) for sigma_val in best_sigma]
			print(best_sigma)
			best_sigma = min(best_sigma)
			print(best_sigma)

			self.sigma = best_sigma
			self._update_sigma(self.sigma, run_context.session)

def serving_input_fn():
	"""Serving function to facilitate model saving."""
	feat = array_ops.placeholder(dtype=dtypes.float32)
	return tf.estimator.export.TensorServingInputReceiver(features=feat,
		receiver_tensors=feat)


def model_fn(features, labels, mode, params):
	""" Main training function ."""

	net = architectures.create_architecture(params)

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

	sample_weights, sample_weights_pos, sample_weights_neg = train_utils.extract_weights(labels, params)
	labels = tf.identity(labels['labels'])

	if mode == tf.estimator.ModeKeys.EVAL:
		main_eval_metrics = {}

		# -- main loss components
		eval_pred_loss, eval_mmd_loss = evaluation_metrics.compute_loss(labels, logits, zpred,
			sample_weights, sample_weights_pos, sample_weights_neg, params)

		main_eval_metrics['pred_loss'] = tf.compat.v1.metrics.mean(eval_pred_loss)
		main_eval_metrics['mmd'] = tf.compat.v1.metrics.mean(eval_mmd_loss)

		loss = eval_pred_loss + params["alpha"] * eval_mmd_loss

		# -- additional eval metrics
		additional_eval_metrics = evaluation_metrics.get_eval_metrics_dict(
			labels, predictions, sample_weights,
			sample_weights_pos, sample_weights_neg, SIGMA_LIST, params)

		eval_metrics = {**main_eval_metrics, **additional_eval_metrics}

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=None, eval_metric_ops=eval_metrics)

	if mode == tf.estimator.ModeKeys.TRAIN:
		gc.collect()
		opt = tf.keras.optimizers.Adam()
		global_step = tf.compat.v1.train.get_global_step()
		sigma_variable = get_or_create_sigma_variable(params['sigma'])

		ckpt = tf.train.Checkpoint(
			step=global_step, optimizer=opt, net=net)

		curr_params = copy.deepcopy(params)
		curr_params['sigma'] = sigma_variable

		with tf.GradientTape() as tape:
			logits, zpred = net(features, training=training_state)
			ypred = tf.nn.sigmoid(logits)

			prediction_loss, mmd_loss = evaluation_metrics.compute_loss(labels, logits, zpred,
				sample_weights, sample_weights_pos, sample_weights_neg, curr_params)

			regularization_loss = tf.reduce_sum(net.losses)
			loss = regularization_loss + prediction_loss + curr_params["alpha"] * mmd_loss


		variables = net.trainable_variables
		gradients = tape.gradient(loss, variables)

		sigma_monitor_hook = tf.estimator.SummarySaverHook(
			save_steps=(params['update_sigma_every_epochs'] * params['steps_per_epoch']) + 1,
			summary_op=[tf.compat.v1.summary.scalar('sigma', sigma_variable)]
		)

		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=tf.group(
				opt.apply_gradients(zip(gradients, variables)),
				ckpt.step.assign_add(1)),
			training_hooks=[sigma_monitor_hook])

def train(exp_dir,
					dataset_builder,
					architecture,
					training_steps,
					pixel,
					num_epochs,
					batch_size,
					Kfolds,
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

	train_utils.cleanup_directory(scratch_exp_dir)

	input_fns = dataset_builder()
	training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater = input_fns
	steps_per_epoch = int(training_data_size / batch_size)

	# TODO: make this a variable
	update_sigma_every_epochs = 2
	save_every_epochs = 2

	params = {
		"pixel": pixel,
		"architecture": architecture,
		"num_epochs": num_epochs,
		"batch_size": batch_size,
		"steps_per_epoch": steps_per_epoch,
		"update_sigma_every_epochs": update_sigma_every_epochs,
		"Kfolds": Kfolds,
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

	# NOTE: need to checkpoint after every epoch
	# for sigma decay to work efficiently
	assert save_every_epochs == update_sigma_every_epochs

	run_config = tf.estimator.RunConfig(
		tf_random_seed=random_seed,
		save_checkpoints_steps=save_every_epochs * steps_per_epoch,
		# save_checkpoints_secs=500,
		keep_checkpoint_max=2)

	est = tf.estimator.Estimator(
		model_fn, model_dir=scratch_exp_dir, params=params, config=run_config)

	print(f"=====steps_per_epoch {steps_per_epoch}======")
	if training_steps == 0:
		training_steps = int(params['num_epochs'] * steps_per_epoch)

		
	est.train(train_input_fn, steps=training_steps,
			hooks=[
			# 	DynamicSigmaHook(
			# 		estimator=est, input_fn_creater=Kfold_input_fn_creater, params=params,
			# 		epoch_size=update_sigma_every_epochs * steps_per_epoch,
			# 		sigma_value=params['sigma']),
				profiler.OomReportingHook()
		],
		saving_listeners=[
			EvalCheckpointSaverListener(est, train_input_fn, "train"),
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.1, params), "0.1"),
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.5, params), "0.5"),
			EvalCheckpointSaverListener(est, eval_input_fn_creater(0.95, params),
				"0.95"),
		]
	)

	validation_results = est.evaluate(valid_input_fn)
	all_results = {"validation": validation_results}

	if py1_y0_shift_list is not None:
		# -- during testing, we dont have access to labels/weights
		test_params = copy.deepcopy(params)
		test_params['weighted_mmd'] = 'False'
		test_params['balanced_weights'] = 'False'
		for py in py1_y0_shift_list:
			eval_input_fn = eval_input_fn_creater(py, test_params)
			distribution_results = est.evaluate(eval_input_fn, steps=1e5)
			all_results[f'shift_{py}'] = distribution_results

	# save results
	savefile = f"{exp_dir}/performance.pkl"
	all_results = train_utils.flatten_dict(all_results)
	pickle.dump(all_results, open(savefile, "wb"))

	# save model
	print(f'{exp_dir}/saved_model')
	est.export_saved_model(f'{exp_dir}/saved_model', serving_input_fn)

	# if cleanup == 'True':
	# 	train_utils.cleanup_directory(scratch_exp_dir)
