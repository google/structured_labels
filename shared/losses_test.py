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

"""Tests for shared.losses."""

import tensorflow as tf

from shared import losses
from absl.testing import absltest

_EMBEDDING_SIZE = 5
_BATCH_SIZE = 10


class LossesTest(absltest.TestCase):
	"""Tests for the MMD losses. """

	def setUp(self):
		super(LossesTest, self).setUp()
		self._embeddings = tf.ones([_BATCH_SIZE, _EMBEDDING_SIZE])
		self._aux_labels = tf.keras.backend.random_bernoulli(
			shape=(_BATCH_SIZE, 1), p=0.5, seed=0)
		self._main_labels = tf.keras.backend.random_bernoulli(
			shape=(_BATCH_SIZE, 1), p=0.5, seed=0)


	def test_identical_embed_mmd(self):
		mmd_val = losses.mmd_loss_unweighted(self._embeddings, self._aux_labels, sigma=1.0)
		self.assertEqual(mmd_val, 0)

	def test_identical_embed_weighted_mmd(self):
		mmd_val = losses.mmd_loss_weighted(self._embeddings, self._main_labels,
			self._aux_labels, sigma=1.0)
		self.assertEqual(mmd_val, 0)


if __name__ == '__main__':
	absltest.main()
