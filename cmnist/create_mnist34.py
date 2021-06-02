""" Creates a local copy of mnist that only has 3's and 4's. """
import os
import pickle

from absl import app
import tensorflow as tf
import numpy as np


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
	'cmnist/data'))


def main(args):
	del args
	if not os.path.exists(DATA_DIR):
		os.system(f'mkdir -p {DATA_DIR}')

	train_data, test_data = tf.keras.datasets.mnist.load_data()

	x_train, y_train = train_data
	keeps_train = (y_train == 3) | (y_train == 4)
	x_train, y_train = x_train[keeps_train].copy(), y_train[keeps_train].copy()
	x_train = x_train[..., np.newaxis] / 255.0
	x_train, y_train = np.float32(x_train), np.float32(y_train)
	pickle.dump((x_train, y_train), open(f'{DATA_DIR}/train_data.pkl', 'wb'))

	x_test, y_test = test_data
	keeps_test = (y_test == 3) | (y_test == 4)
	x_test, y_test = x_test[keeps_test].copy(), y_test[keeps_test].copy()
	x_test = x_test[..., np.newaxis] / 255.0
	x_test, y_test = np.float32(x_test), np.float32(y_test)
	print(x_test.shape)
	pickle.dump((x_test, y_test), open(f'{DATA_DIR}/test_data.pkl', 'wb'))


if __name__ == '__main__':
	app.run(main)
