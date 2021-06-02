import numpy as np
import train as train

N = 1000
D = 5

if __name__ == "__main__":
	np.random.seed(1)
	x = np.vstack([
		np.random.normal(0, 1, (int(N / 2), D)),
		np.random.normal(10, 1, (int(N / 2), D))
	])

	y1 = np.vstack([
		np.random.binomial(1, 0.9, (int(N / 2), 1)),
		np.random.binomial(1, 0.1, (int(N / 2), 1))
	])

	y0 = np.random.binomial(1, 0.5, (N, 1))
	yh = y0*0.9 + (1-y0)*0.1
	labels = np.hstack([y0, y1])

	model = train.KernelSlabs(kernel='rbf', sigma=10.0, alpha=0.0, C=0.1)
	model.fit(x, labels)

	yp = model.predict(x)
	err = np.abs(yp - yh)
	print(np.mean(err))

