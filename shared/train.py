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

from shared import architectures
from shared import losses
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def flatten_dict(dd, separator='_', prefix=''):
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


def model_fn(features, labels, mode, params):
	"""Main training function ."""
	net = architectures.SimpleConvolutionNet(
		dropout_rate=params["dropout_rate"],
		l2_penalty=params["l2_penalty"],
		embedding_dim=params["embedding_dim"])

	if mode == tf.estimator.ModeKeys.PREDICT:
		logits, _ = net(features)
		ypred = tf.nn.sigmoid(logits)

		predictions = {
			"classes": tf.math.greater_equal(ypred, .5),
			"probabilities": ypred
		}

		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				"classify": tf.estimator.export.PredictOutput(predictions)
			})

	if mode == tf.estimator.ModeKeys.EVAL:
		logits, zpred = net(features)
		ypred = tf.nn.sigmoid(logits)

		predictions = {
			"classes": tf.math.greater_equal(ypred, .5),
			"probabilities": ypred
		}

		y_aug = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

		individual_losses = tf.keras.losses.binary_crossentropy(
			y_aug, logits, from_logits=True)
		ypred_loss = tf.reduce_mean(individual_losses)
		loss = ypred_loss

		other_label_inds = [
			lab_ind for lab_ind in range(labels.shape[1])
			if lab_ind != params["label_ind"]
		]

		all_mmd_vals = []
		for lab_ind in other_label_inds:
			mmd_val = losses.mmd_loss(
				embedding=zpred,
				main_labels=labels[:, params["label_ind"]],
				auxiliary_labels=labels[:, lab_ind],
				sigma=params["sigma"],
				weighted=params["weighted_mmd"])
			all_mmd_vals.append(mmd_val[0])

		all_mmds = tf.concat(all_mmd_vals, axis=0)

		eval_metrics = {
			"accuracy":
				tf.compat.v1.metrics.accuracy(
					labels=y_aug, predictions=predictions["classes"]),
			"mmd": tf.compat.v1.metrics.mean(all_mmds)
		}

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=None, eval_metric_ops=eval_metrics)

	if mode == tf.estimator.ModeKeys.TRAIN:
		opt = tf.keras.optimizers.Adam()
		ckpt = tf.train.Checkpoint(
			step=tf.compat.v1.train.get_global_step(), optimizer=opt, net=net)
		with tf.GradientTape() as tape:
			logits, zpred = net(features, training=True)
			# prediction loss
			y_aug = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

			individual_losses = tf.keras.losses.binary_crossentropy(
				y_aug, logits, from_logits=True)
			ypred_loss = tf.reduce_mean(individual_losses)
			# mmd
			other_label_inds = [
				lab_ind for lab_ind in range(labels.shape[1])
				if lab_ind != params["label_ind"]
			]

			all_mmd_vals = []
			for lab_ind in other_label_inds:
				mmd_val = losses.mmd_loss(
					embedding=zpred,
					main_labels=labels[:, params["label_ind"]],
					auxiliary_labels=labels[:, lab_ind],
					sigma=params["sigma"],
					weighted=params["weighted_mmd"])
				all_mmd_vals.append(mmd_val[0])

			all_mmds = tf.concat(all_mmd_vals, axis=0)
			mmd_loss_val = tf.reduce_sum(all_mmds)
			regularization_loss = tf.reduce_sum(net.losses)
			loss = ypred_loss + params["alpha"] * mmd_loss_val + regularization_loss

		variables = net.trainable_variables
		gradients = tape.gradient(loss, variables)

		# summary_hook = tf.estimator.SummarySaverHook(save_steps=10,
		# 	summary_op=[
		# 		tf.compat.v1.summary.scalar('mmd1', mmd_val[1]),
		# 		tf.compat.v1.summary.scalar('mmd0', mmd_val[2]),
		# 		tf.compat.v1.summary.scalar('mmd10', mmd_val[3]),
		# 		tf.compat.v1.summary.scalar('mmd01', mmd_val[4])
		# 	])

		logging_hook = tf.estimator.LoggingTensorHook(every_n_iter=10,
			tensors={
				'mmd1': mmd_val[1],
				'mmd0': mmd_val[2],
				'mmd10': mmd_val[3],
				'mmd01': mmd_val[4]
			})

		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=tf.group(
				opt.apply_gradients(zip(gradients, variables)),
				ckpt.step.assign_add(1)),
			training_hooks=[logging_hook])

def train(exp_dir,
					dataset_builder,
					training_steps,
					num_epochs,
					batch_size,
					alpha,
					sigma,
					weighted_mmd,
					dropout_rate,
					l2_penalty,
					embedding_dim,
					random_seed,
					cleanup,
					py1_y0_shift_list=None):
	"""Trains the estimator."""

	if not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	params = {
		"batch_size": batch_size,
		"num_epochs": num_epochs,
		"alpha": alpha,
		"sigma": sigma,
		"weighted_mmd": weighted_mmd,
		"dropout_rate": dropout_rate,
		"l2_penalty": l2_penalty,
		"embedding_dim": embedding_dim,
		"label_ind": 0
	}

	run_config = tf.estimator.RunConfig(
		tf_random_seed=random_seed,
		save_checkpoints_secs=300,
		keep_checkpoint_max=2)

	est = tf.estimator.Estimator(
		model_fn, model_dir=exp_dir, params=params, config=run_config)

	input_fns = dataset_builder()
	train_input_fn, valid_input_fn, eval_input_fn_creater = input_fns
	# Do the actual training.
	est.train(train_input_fn, steps=training_steps)
	validation_results = est.evaluate(valid_input_fn)
	all_results = {"validation": validation_results}

	if py1_y0_shift_list is not None:
		for py in py1_y0_shift_list:
			eval_input_fn = eval_input_fn_creater(py)
			distribution_results = est.evaluate(eval_input_fn)
			all_results[f'shift_{py}'] = distribution_results

	if cleanup:
		cleanup_directory(exp_dir)

	savefile = f"{exp_dir}/performance.pkl"
	all_results = flatten_dict(all_results)
	pickle.dump(all_results, open(savefile, "wb"))
