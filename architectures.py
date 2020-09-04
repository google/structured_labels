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

"""Commonly used neural network architectures."""

import tensorflow as tf


class SimpleConvolutionNet(tf.keras.Model):
  """Simple architecture with convolutions + max pooling."""

  def __init__(self, dropout_rate=0.0, l2_penalty=0.0, embedding_dim=1000):
    super(SimpleConvolutionNet, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, [3, 3], activation="relu")
    self.conv2 = tf.keras.layers.Conv2D(64, [3, 3], activation="relu")
    self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    self.flatten1 = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        embedding_dim,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2_penalty),
        name="Z")
    self.dense2 = tf.keras.layers.Dense(1)

  def call(self, inputs, training=False):
    z = self.conv1(inputs)
    z = self.conv2(z)
    z = self.maxpool1(z)
    if training:
      z = self.dropout(z, training=training)
    z = self.flatten1(z)
    z = self.dense1(z)
    return self.dense2(z), z
