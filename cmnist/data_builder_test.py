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

"""Tests for structured_labels.data_builder."""

from absl.testing import absltest
import numpy as np
from cmnist import data_builder


class CmnistLibTest(absltest.TestCase):
  # generate dummy data that looks like mnist
  y = np.random.choice(range(10), size=5000, replace=True)
  x = np.zeros((5000, 28, 28, 1))
  x[:, 0, :, 0] = np.random.uniform(0, 1, (5000, 28))

  # dummy image
  img = x[0, :, :, :]

  def test_color_corrupt_1(self):
    # if y == 1 there should be no pixels > .5 in channel 2
    xc1 = data_builder.color_corrupt_img(self.img, 1, 1e5)
    xc1_mask = xc1[:, :, 2] > .5
    correct_mask = np.zeros_like(xc1_mask, dtype=bool)
    incorr = np.sum(xc1_mask != correct_mask)
    self.assertEqual(incorr, 0)

  def test_color_corrupt_0(self):
    # if y == 0 there should be no pixels > .5 in channel 1
    xc0 = data_builder.color_corrupt_img(self.img, 0, 1e5)
    xc0_mask = xc0[:, :, 1] > .5
    correct_mask = np.zeros_like(xc0_mask, dtype=bool)
    incorr = np.sum(xc0_mask != correct_mask)
    self.assertEqual(incorr, 0)

  def test_corrupt_mnist_noiseless_1(self):
    # (no noise) if py1_y0 = 1 the two outcomes should be same
    _, y_lab = data_builder.corrupt_mnist(
        self.x, self.y, py1_y0=1, pflip0=0, pflip1=0,
        npix=5)

    diff = np.sum(np.abs(y_lab[:, 0] - y_lab[:, 1]))
    self.assertEqual(diff, 0)

  def test_corrupt_mnist_noisy_1(self):
    # (noisy) if py1_y0 = 1 at most pflip should be different
    _, y_lab = data_builder.corrupt_mnist(
        self.x, self.y, py1_y0=1, pflip0=.1, pflip1=0,
        npix=5)

    p_disagree = np.mean(y_lab[:, 0] != y_lab[:, 1])
    self.assertLessEqual(p_disagree, .1)

  def test_corrupt_mnist_noiseless_0(self):
    _, y_lab = data_builder.corrupt_mnist(
        self.x, self.y, py1_y0=0, pflip0=0, pflip1=0,
        npix=5)

    diff = np.sum(np.abs(y_lab[:, 0] - y_lab[:, 1]))
    self.assertEqual(diff, y_lab.shape[0])

  def test_corrupt_mnist_noisy_0(self):
    _, y_lab = data_builder.corrupt_mnist(
        self.x, self.y, py1_y0=0, pflip0=.1, pflip1=0,
        npix=5)

    p_agree = np.mean(y_lab[:, 0] == y_lab[:, 1])
    self.assertLessEqual(p_agree, .1)


if __name__ == '__main__':
  absltest.main()
