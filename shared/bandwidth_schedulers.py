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

"""Schedulers for dynamically setting the MMD bandwidth during training."""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops, dtypes

import pandas as pd


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


class ExponentialDecaySigmaScheduler(tf.estimator.SessionRunHook):
	"""Decays sigma exponentially after every epoch"""

	def __init__(self, decay_rate, epoch_size, sigma_value):
		self.decay_rate = decay_rate
		self.sigma = sigma_value
		self.epoch_size = epoch_size
		self.step = 0

	def begin(self):
		self._global_step_tensor = tf.compat.v1.train.get_global_step()
		self.variable = get_or_create_sigma_variable(self.sigma)

	def _update_sigma(self, value, session):
		return self.variable.load(value, session=session)

	def after_run(self, run_context, run_values):
		del run_values
		current_global_step_value = run_context.session.run(self._global_step_tensor)
		if current_global_step_value % self.epoch_size == 0:  # epoch
			self.step += 1.0
			self.sigma = self.sigma * (1.0 - self.decay_rate) ** self.step
			self._update_sigma(self.sigma, run_context.session)


class AdaptiveSigmaScheduler(tf.estimator.SessionRunHook):
	"""Sets sigma based on an external validation set"""

	def __init__(self, estimator, input_fn_creater, params, epoch_size,
		sigma_value):
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
		del run_values
		current_global_step_value = run_context.session.run(self._global_step_tensor)
		if current_global_step_value % self.epoch_size == 0:  # epoch
			results = []
			for foldid in range(self.params['Kfolds']):
				fold_results = self.estimator.evaluate(self.input_fn_creater(foldid),
					name=f'bandwidth_selection_{foldid}')
				if foldid == 0:
					print(f' ########## Changing sigma at global step {current_global_step_value} ###########')
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