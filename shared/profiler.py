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

"""Profiler for estimator."""
import os
import tensorflow as tf

from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline


class OomReportingHook(tf.estimator.SessionRunHook):
	def before_run(self, run_context):
		return SessionRunArgs(fetches=[],  # no extra fetches
													options=tf.compat.v1.RunOptions(
														report_tensor_allocations_upon_oom=True))


class MetadataHook(tf.estimator.SessionRunHook):
	def __init__(self, save_steps=None, save_secs=None, output_dir=""):
		self._output_tag = "step-{}"
		self._output_dir = output_dir
		self._timer = SecondOrStepTimer(
			every_secs=save_secs, every_steps=save_steps)

	def begin(self):
		self._next_step = None
		self._global_step_tensor = training_util.get_global_step()
		self._writer = tf.compat.v1.summary.FileWriter(
			self._output_dir, tf.compat.v1.get_default_graph())

		if self._global_step_tensor is None:
			raise RuntimeError("Global step should be created to use ProfilerHook.")

	def before_run(self, run_context):
		del run_context
		self._request_summary = (
			self._next_step is None or
			self._timer.should_trigger_for_step(self._next_step)
		)
		requests = {"global_step": self._global_step_tensor}
		opts = (tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
				if self._request_summary else None)

		# opts = (
		# 	config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
		# 	if self._request_summary else None)

		return SessionRunArgs(requests, options=opts)

	def after_run(self, run_context, run_values):
		stale_global_step = run_values.results["global_step"]
		global_step = stale_global_step + 1
		if self._request_summary:
			global_step = run_context.session.run(self._global_step_tensor)
			self._writer.add_run_metadata(
				run_values.run_metadata, self._output_tag.format(global_step))
			self._writer.flush()
		self._next_step = global_step + 1

	def end(self, session):
		del session
		self._writer.close()


class EagerProfilerHook(tf.estimator.SessionRunHook):
	"""Main class for profiler. based on
	https://github.com/tensorflow/tensorflow/issues/27545
	"""

	def __init__(self,
							save_steps=None,
							save_secs=None,
							output_dir="",
							show_dataflow=True,
							show_memory=False):
		"""Initializes a hook that takes periodic profiling snapshots.
		`options.run_metadata` argument of `tf.Session.Run` is used to collect
		metadata about execution. This hook sets the metadata and dumps it in Chrome
		Trace format.
		Args:
			save_steps: `int`, save profile traces every N steps. Exactly one of
				`save_secs` and `save_steps` should be set.
			save_secs: `int` or `float`, save profile traces every N seconds.
			output_dir: `string`, the directory to save the profile traces to.
				Defaults to the current directory.
			show_dataflow: `bool`, if True, add flow events to the trace connecting
				producers and consumers of tensors.
			show_memory: `bool`, if True, add object snapshot events to the trace
				showing the sizes and lifetimes of tensors.
		"""
		self._output_file = os.path.join(output_dir, "timeline-{}.json")
		self._file_writer = tf.summary.create_file_writer(output_dir,
			name='profiler')
		self._show_dataflow = show_dataflow
		self._show_memory = show_memory
		self._timer = SecondOrStepTimer(
			every_secs=save_secs, every_steps=save_steps)

	def begin(self):
		self._next_step = None
		self._global_step_tensor = training_util._get_or_create_global_step_read()
		if self._global_step_tensor is None:
			raise RuntimeError("Global step should be created to use ProfilerHook.")

	def before_run(self, run_context):
		del run_context
		self._request_summary = (
			self._next_step is not None and
			self._timer.should_trigger_for_step(self._next_step))
		requests = {"global_step": self._global_step_tensor}
		opts = (
			config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
			if self._request_summary else None)

		return SessionRunArgs(requests, options=opts)

	def after_run(self, run_context, run_values):
		stale_global_step = run_values.results["global_step"]
		if self._next_step is None:
			# Update the timer so that it does not activate until N steps or seconds
			# have passed.
			self._timer.update_last_triggered_step(stale_global_step)
		global_step = stale_global_step + 1
		if self._request_summary:
			global_step = run_context.session.run(self._global_step_tensor)
			self._timer.update_last_triggered_step(global_step)
			self._save(global_step, self._output_file.format(global_step),
				run_values.run_metadata.step_stats)

		self._next_step = global_step + 1

	def _save(self, step, save_path, step_stats):
		logging.info("Saving timeline for %d into '%s'.", step, save_path)
		with gfile.Open(save_path, "w") as f:
			trace = timeline.Timeline(step_stats)
			f.write(
				trace.generate_chrome_trace_format(
					show_dataflow=self._show_dataflow, show_memory=self._show_memory))
