# Copyright 2020 the Causally Motivated Shortcut Removal
# Authors. All rights reserved.

"""Library of neural network architectures used."""


import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.keras.applications.resnet50 import ResNet50

def create_architecture(params):

	if params['architecture'] == 'simple':
		net = SimpleConvolutionNet(
			dropout_rate=params["dropout_rate"],
			l2_penalty=params["l2_penalty"],
			embedding_dim=params["embedding_dim"])

	elif params['architecture'] == 'pretrained_resnet' and (params['random_augmentation'] == "False"):
		net = PretrainedResNet50(
			l2_penalty=params["l2_penalty"])

	elif params['architecture'] == 'resnet_random':
		net = RandomResNet50(
			l2_penalty=params["l2_penalty"])

	elif (params['architecture'] == 'pretrained_resnet') and (params['random_augmentation'] == "True"):
		net = PretrainedResNet50_RandomAugmentation(
			l2_penalty=params["l2_penalty"])

	return net

class PretrainedResNet50(tf.keras.Model):
	"""Resnet50 pretrained on imagenet."""

	def __init__(self, l2_penalty=0.0, l2_penalty_last_only=False):
		super(PretrainedResNet50, self).__init__()

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		return self.dense(x), x


class PretrainedResNet50_RandomAugmentation(tf.keras.Model):
	"""Resnet50 pretrained on imagenet with random augmentations"""

	def __init__(self, l2_penalty=0.0, l2_penalty_last_only=False):
		super(PretrainedResNet50_RandomAugmentation, self).__init__()

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
		return self.dense(x), x

class RandomResNet50(tf.keras.Model):
	"""Randomly initialized resnet 50."""

	def __init__(self, l2_penalty=0.0):
		super(RandomResNet50, self).__init__()
		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights=None)
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		regularizer = tf.keras.regularizers.l2(l2_penalty)
		for layer in self.resenet.layers:
			if hasattr(layer, 'kernel'):
				self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
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

