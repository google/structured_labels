""" Main training module for kernel version"""
import numpy as np
import cvxpy as cp

from sklearn.gaussian_process.kernels import RBF


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def mmd_loss_method2(x, auxiliary_labels, sigma):

	x1 = x[auxiliary_labels == 1]
	x0 = x[auxiliary_labels == 0]

	pos_kernel = RBF(length_scale=sigma).__call__(x1, x1)
	neg_kernel = RBF(length_scale=sigma).__call__(x0, x0)
	pos_neg_kernel = RBF(length_scale=sigma).__call__(x1, x0)

	pos_kernel_mean = np.mean(pos_kernel)
	neg_kernel_mean = np.mean(neg_kernel)
	pos_neg_kernel_mean = np.mean(pos_neg_kernel)

	mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
	print(pos_kernel_mean)
	print(neg_kernel_mean)
	print(pos_neg_kernel_mean)
	mmd_val = np.maximum(0.0, mmd_val)
	return mmd_val

def mmd_loss(kernel, auxiliary_labels):

	auxiliary_labels = np.expand_dims(auxiliary_labels, 1)

	pos_mask = auxiliary_labels * auxiliary_labels.transpose()
	neg_mask = (1.0 - auxiliary_labels) * (1.0 - auxiliary_labels).transpose()
	pos_neg_mask = auxiliary_labels * (1.0 - auxiliary_labels).transpose()

	if np.sum(pos_mask) == 0:
		pos_kernel_mean = 0
	else:
		pos_kernel_mean = np.sum(pos_mask * kernel) / np.sum(pos_mask)

	if np.sum(neg_mask) == 0:
		neg_kernel_mean = 0
	else:
		neg_kernel_mean = np.sum(neg_mask * kernel) / np.sum(neg_mask)

	if np.sum(pos_neg_mask) == 0:
		pos_neg_kernel_mean = 0
	else:
		pos_neg_kernel_mean = np.sum(pos_neg_mask * kernel) / np.sum(pos_neg_mask)

	mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
	print(pos_kernel_mean)
	print(neg_kernel_mean)
	print(pos_neg_kernel_mean)
	mmd_val = np.maximum(0.0, mmd_val)
	print(mmd_val)
	return mmd_val


class KernelSlabs():
	""" Kernel version of our model """

	def __init__(self, kernel, sigma, C, **unused_kwargs):

		""" Constructor """
		if ((kernel != 'rbf') and (kernel != 'poly')):
			raise NotImplementedError("Incorrect kernel string. Pick from rbf and poly")
		self.kernel = kernel
		self.sigma = sigma
		self.C = C

	def fit(self, x, labels, w=None):
		""" Fits the main function """

		# -- fit kernel
		if self.kernel == 'rbf':
			self.x_tr = x.copy()
			self.kernel_fit = lambda x_ts: RBF(length_scale=self.sigma).__call__(x_ts,
				self.x_tr)
		elif self.kernel == 'poly':
			self.kernel_fit = lambda x: np.hstack(
				[x**i for i in range(1, int(self.sigma) + 1)])

		x = self.kernel_fit(x)
		y = labels[:, 0]

		mmd_val = mmd_loss(x, labels[:, 1])

		# -- setup opt problem
		n, d = x.shape[0], x.shape[1]
		if w is None:
			w = np.ones(n) / n

		bf = cp.Variable(d)
		individual_log_like = cp.multiply(y, x @ bf) - cp.logistic(x @ bf)
		log_likelihood = cp.sum(cp.multiply(w, individual_log_like))
		problem = cp.Problem(cp.Maximize(log_likelihood - self.C * cp.norm(bf, 2)))
		problem.solve()

		self.bf = bf.value

	def predict(self, x):
		""" Predicts lower and upper bounds at a point x """
		x = self.kernel_fit(x)

		yh = np.dot(x, self.bf)
		yh = sigmoid(yh)

		return yh

