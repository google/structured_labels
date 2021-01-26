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

# NOTE:see batch norm issues here https://github.com/keras-team/keras/pull/9965
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.keras.applications.resnet50 import ResNet50

# from tensorflow.keras.layers.experimental import preprocessing


def create_architecture(params):

	if params['architecture'] == 'simple':
		net = SimpleConvolutionNet(
			dropout_rate=params["dropout_rate"],
			l2_penalty=params["l2_penalty"],
			embedding_dim=params["embedding_dim"])
	elif params['architecture'] == 'pretrained_resnet' and (params['random_augmentation'] == "False"):

		net = PretrainedResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])
	elif params['architecture'] == 'pretrained_resnet_random':
		net = RandomResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])

	elif (params['architecture'] == 'pretrained_resnet') and (params['random_augmentation'] == "True"):
		net = PretrainedResNet50_RandomAugmentation(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])

	elif params['architecture'] == 'pretrained_resnet101':
		net = PretrainedResNet101(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])

	elif params['architecture'] == 'from_scratch_resnet':
		net = ScratchResNet50()

	return net

class ResnetIdentityBlock(tf.keras.Model):
	def __init__(self, filters, kernel_size=3, stride=1, name=None):
		# check naming
		super(ResnetIdentityBlock, self).__init__()
		bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

		self.conv1 = tf.keras.layers.Conv2D(filters, 1, strides=stride,
			name=name + '_1_conv')
		self.bn1 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')
		self.relu1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')

		self.conv2 = tf.keras.layers.Conv2D(
			filters, kernel_size, padding='SAME', name=name + '_2_conv')
		self.bn2 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')
		self.relu2 = tf.keras.layers.Activation('relu', name=name + '_2_relu')

		self.conv3 = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')
		self.bn3 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')

		self.add = tf.keras.layers.Add(name=name + '_add') #([shortcut, x])
		self.relu_out = tf.keras.layers.Activation('relu', name=name + '_out')

	@tf.function
	def call(self, input_tensor, training=False):
		x = self.conv1(input_tensor)
		x = self.bn1(x, training=training)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x = self.relu2(x)

		x = self.conv3(x)
		x = self.bn3(x, training=training)

		x = self.add([input_tensor, x])
		x = self.relu_out(x)
		return x

class ResnetConvBlock(tf.keras.Model):
	def __init__(self, filters, kernel_size=3, stride=1, name=None):
		# check naming
		super(ResnetConvBlock, self).__init__()
		bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

		self.conv0 = tf.keras.layers.Conv2D(
			4 * filters, 1, strides=stride, name=name + '_0_conv')
		self.bn0 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')

		self.conv1 = tf.keras.layers.Conv2D(filters, 1, strides=stride,
			name=name + '_1_conv')
		self.bn1 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')
		self.relu1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')

		self.conv2 = tf.keras.layers.Conv2D(
			filters, kernel_size, padding='SAME', name=name + '_2_conv')
		self.bn2 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')
		self.relu2 = tf.keras.layers.Activation('relu', name=name + '_2_relu')

		self.conv3 = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')
		self.bn3 = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')

		self.add = tf.keras.layers.Add(name=name + '_add') #([shortcut, x])
		self.relu_out = tf.keras.layers.Activation('relu', name=name + '_out')

	@tf.function
	def call(self, input_tensor, training=False):
		x_conv = self.conv0(input_tensor)
		x_conv = self.bn0(x_conv, training=training)

		x = self.conv1(input_tensor)
		x = self.bn1(x, training=training)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x = self.relu2(x)

		x = self.conv3(x)
		x = self.bn3(x, training=training)

		x = self.add([x_conv, x])
		x = self.relu_out(x)
		return x


class ResnetStack(tf.keras.Model):
	def __init__(self, filters, blocks, stride=1, name=None):
		# check naming
		super(ResnetStack, self).__init__()
		self.blocks = blocks

		self.block1 = ResnetConvBlock(filters=filters, stride=stride,
			name=name + '_block1')
		self.block2 = ResnetIdentityBlock(filters=filters, stride=stride,
			name=name + '_block2')
		self.block3 = ResnetIdentityBlock(filters=filters, stride=stride,
			name=name + '_block3')

		if blocks > 3:
			self.block4 = ResnetIdentityBlock(filters=filters, stride=stride,
				name=name + '_block4')

		if blocks == 6:
			self.block5 = ResnetIdentityBlock(filters=filters, stride=stride,
				name=name + '_block5')
			self.block6 = ResnetIdentityBlock(filters=filters, stride=stride,
				name=name + '_block6')

	@tf.function
	def call(self, input_tensor, training=False):
		x = self.block1(input_tensor, training)
		x = self.block2(x, training)
		x = self.block3(x, training)

		if self.blocks > 3:
			x = self.block4(x, training)
		if self.blocks == 6:
			x = self.block5(x, training)
			x = self.block6(x, training)
		return x


class ScratchResNet50(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self):
		super(ScratchResNet50, self).__init__()

		bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

		self.conv1_pad = tf.keras.layers.ZeroPadding2D(
			padding=((3, 3), (3, 3)), name='conv1_pad')
		self.conv1_conv = tf.keras.layers.Conv2D(64, 7,
			strides=2, use_bias=True, name='conv1_conv')
		self.conv1_bn = tf.keras.layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')
		self.conv1_relu = tf.keras.layers.Activation('relu',
			name='conv1_relu')

		self.pool1_pad = tf.keras.layers.ZeroPadding2D(
			padding=((1, 1), (1, 1)), name='pool1_pad')
		self.pool1_pool = tf.keras.layers.MaxPooling2D(3,
			strides=2, name='pool1_pool')

		self.stack1 = ResnetStack(64, 3, stride=1, name='conv2')
		self.stack2 = ResnetStack(128, 4, name='conv3')
		self.stack3 = ResnetStack(256, 6, name='conv4')
		self.stack4 = ResnetStack(512, 3, name='conv5')

		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
		self.dense = tf.keras.layers.Dense(1)

	@tf.function
	def call(self, inputs, training=False):

		x = self.conv1_pad(inputs)
		x = self.conv1_conv(x)
		x = self.conv1_bn(x, training)
		x = self.conv1_relu(x)
		x = self.pool1_pad(x)
		x = self.pool1_pool(x)
		x = self.stack1(x, training)
		x = self.stack2(x, training)
		x = self.stack3(x, training)
		x = self.stack4(x, training)
		x = self.avg_pool(x)
		return self.dense(x), x


class PretrainedResNet50(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=10, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedResNet50, self).__init__()
		self.embedding_dim = embedding_dim

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		# TODO fix this
		if self.embedding_dim != 10:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != 10:
			x = self.embedding(x)
		return self.dense(x), x


class PretrainedResNet50_RandomAugmentation(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=10, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedResNet50_RandomAugmentation, self).__init__()
		self.embedding_dim = embedding_dim

		self.data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
		])

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		# TODO fix this
		if self.embedding_dim != 10:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		if training:
			x = self.data_augmentation(inputs, training=training)
		else: 
			x = inputs
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != 10:
			x = self.embedding(x)
		return self.dense(x), x

class RandomResNet50(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=10, l2_penalty=0.0):
		super(RandomResNet50, self).__init__()
		self.embedding_dim = embedding_dim
		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights=None)
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		regularizer = tf.keras.regularizers.l2(l2_penalty)
		for layer in self.resenet.layers:
			if hasattr(layer, 'kernel'):
				self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		# TODO fix this
		if self.embedding_dim != 10:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != 10:
			x = self.embedding(x)
		return self.dense(x), x

class PretrainedResNet101(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=10, l2_penalty=0.0):
		super(PretrainedResNet101, self).__init__()
		self.embedding_dim = embedding_dim
		self.resenet = tf.keras.applications.ResNet101(include_top=False,
			layers=tf.keras.layers, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		regularizer = tf.keras.regularizers.l2(l2_penalty)
		for layer in self.resenet.layers:
			if hasattr(layer, 'kernel'):
				self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		# TODO fix this
		if self.embedding_dim != 10:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != 10:
			x = self.embedding(x)
		return self.dense(x), x


class SimpleConvolutionNet(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, dropout_rate=0.0, l2_penalty=0.0, embedding_dim=1000):
		super(SimpleConvolutionNet, self).__init__()
		# self.scale = preprocessing.Rescaling(1.0 / 255)
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

