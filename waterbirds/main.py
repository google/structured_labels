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

"""Main file for running the corrupted mnist experiment."""

from absl import app
from absl import flags

from waterbirds import data_builder
from shared import train
from shared.utils import restrict_GPU_tf

FLAGS = flags.FLAGS
flags.DEFINE_float('p_tr', .8, 'proportion of data used for training.')
flags.DEFINE_float('py0', .8, 'Probability of water bird = 1.')
flags.DEFINE_float('py1_y0', .95, '(unshifted) probability of y1 =1 | y0 = 1.')
flags.DEFINE_list('py1_y0_shift_list', [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .1],
	'(shifted) probability of y1 =1 | y0 = 1.')
flags.DEFINE_float('pflip0', 0.01, 'proportion of y0 randomly flipped (noise).')
flags.DEFINE_float('pflip1', 0.01, 'proportion of y1 randomly flipped (noise).')
flags.DEFINE_float('oracle_prop', 0.0,
										'proportion of training data to use for oracle augmentation.')


flags.DEFINE_string('exp_dir', '/data/ddmg/slabs/dummy/',
										'Directory to save trained model in.')
flags.DEFINE_string('architecture', 'pretrained_resnet',
										'Architecture to use for training.')
flags.DEFINE_integer('pixel', 64, 'number of pixels in the image (i.e., res).')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_integer('training_steps', 11050,
										'number of estimator training steps.')
flags.DEFINE_float('alpha', 1.0, 'Value for the cross prediction penelty')
flags.DEFINE_float('sigma', 1.0, 'Value for the MMD kernel bandwidth.')
flags.DEFINE_string('weighted_mmd', 'False',
											'use weighting when computing the mmd?.')
flags.DEFINE_string('balanced_weights', 'True',
											'balance weights? aka add numerator.')

flags.DEFINE_float('dropout_rate', 0.0, 'Value for drop out rate')
flags.DEFINE_float('l2_penalty', 0.0,
									'L2 regularization penalty for final layer')
flags.DEFINE_integer('embedding_dim', 1000,
										'Dimension for the final embedding.')
flags.DEFINE_integer('random_seed', 0, 'random seed for tensorflow estimator')
flags.DEFINE_boolean('minimize_logits', False,
		'compute mmd wrt to logits if true and embedding if false.')

flags.DEFINE_boolean('cleanup', False,
		'remove tensorflow artifacts after training to reduce memory usage.')
flags.DEFINE_string('gpuid', '0', 'Gpu id to run the model on.')


def main(argv):

	del argv

	if isinstance(FLAGS.py1_y0_shift_list[0], str):
		py1_y0_shift_list = [float(val) for val in FLAGS.py1_y0_shift_list]
	else:
		py1_y0_shift_list = FLAGS.py1_y0_shift_list

	def dataset_builder():
		return data_builder.build_input_fns(
			p_tr=FLAGS.p_tr,
			py0=FLAGS.py0,
			py1_y0=FLAGS.py1_y0,
			py1_y0_s=py1_y0_shift_list,
			pflip0=FLAGS.pflip0,
			pflip1=FLAGS.pflip1,
			oracle_prop=FLAGS.oracle_prop,
			random_seed=FLAGS.random_seed)

	restrict_GPU_tf(FLAGS.gpuid)

	train.train(
		exp_dir=FLAGS.exp_dir,
		dataset_builder=dataset_builder,
		architecture=FLAGS.architecture,
		training_steps=FLAGS.training_steps,
		pixel=FLAGS.pixel,
		batch_size=FLAGS.batch_size,
		alpha=FLAGS.alpha,
		sigma=FLAGS.sigma,
		balanced_weights=FLAGS.balanced_weights,
		weighted_mmd=FLAGS.weighted_mmd,
		dropout_rate=FLAGS.dropout_rate,
		l2_penalty=FLAGS.l2_penalty,
		embedding_dim=FLAGS.embedding_dim,
		random_seed=FLAGS.random_seed,
		minimize_logits=FLAGS.minimize_logits,
		cleanup=FLAGS.cleanup,
		py1_y0_shift_list=py1_y0_shift_list)


if __name__ == '__main__':
	app.run(main)
