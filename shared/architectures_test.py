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

"""Tests for shared.architectures."""

from absl.testing import absltest
from shared import architectures
import tensorflow as tf

_BATCH_SIZE = 10
_INPUT_SIZE = 28
_INPUT_CHANNELS = 3
_EMBEDDING_DIM = 128
_L2_PENALTY = 0.0
_DROPOUT_RATE = 0.0


class ArchitecturesTest(absltest.TestCase):

  def setUp(self):
    super(ArchitecturesTest, self).setUp()
    self._inputs = tf.ones(
        [_BATCH_SIZE, _INPUT_SIZE, _INPUT_SIZE, _INPUT_CHANNELS])

  def test_output_shapes(self):
    net = architectures.SimpleConvolutionNet(
        dropout_rate=_DROPOUT_RATE,
        l2_penalty=_L2_PENALTY,
        embedding_dim=_EMBEDDING_DIM)
    y, z = net(self._inputs)
    self.assertEqual(y.shape, (_BATCH_SIZE, 1))
    self.assertEqual(z.shape, (_BATCH_SIZE, _EMBEDDING_DIM))


if __name__ == '__main__':
  absltest.main()
