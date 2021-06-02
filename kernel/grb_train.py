""" Main training module for kernel version"""
import gurobipy as grb
import numpy as np

from sklearn.gaussian_process.kernels import RBF


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class KernelSlabs():
	""" Kernel version of our model """

	def __init__(self, sigma, alpha, C, **unused_kwargs):

		""" Constructor """

		self.sigma = sigma
		self.alpha = alpha
		self.C = C

	def fit(self, x, y, w=None):
		""" Fits the main function """

		# -- fit kernel
		self.x_tr = x.copy()
		self.kernel_fit = lambda x_ts: RBF(length_scale=self.sigma).__call__(x_ts,
			self.x_tr)
		x = self.kernel_fit(x)

		# -- setup opt problem
		n, d = x.shape[0], x.shape[1]
		mdl = grb.Model("qp")
		mdl.ModelSense = 1
		mdl.setParam('OutputFlag', True)
		mdl.reset()

		if w is None:
			w = np.ones(n) / n

		# -- params to learn
		L = 1e5

		fs = [mdl.addVar(name="fs%d" % i, lb=-L, ub=L) for i in range(n)]
		log_fs = [mdl.addVar(name="log_fs%d" % i, lb=-L, ub=L) for i in range(n)]

		bf = [mdl.addVar(name="bf%d" % i, lb=-L, ub=L) for i in range(d + 1)]

		main_loss = []
		for i in range(n):
			mdl.addConstr(fs[i] == np.dot(x[i, ], bf[:d]) + bf[-1])
			mdl.addGenConstrPWL(log_fs[i] == np.log(1.0 + np.exp(fs[i])))
			main_loss.append(w[i] * (- (y[i] * fs[i]) + log_fs[i]))

		obj = grb.quicksum(main_loss)

		mdl.setObjective(obj)
		mdl.optimize()

		self.bf = np.array([bf[j].x for j in range(d+1)])
		# print(obj.getValue(), obj_slack.getValue())

		return self

	def predict(self, x):
		""" Predicts lower and upper bounds at a point x """
		x = self.kernel_fit(x)

		yh = np.dot(x, self.bf[:x.shape[1]]) + self.bf[-1]
		yh = sigmoid(yh)

		return yh
