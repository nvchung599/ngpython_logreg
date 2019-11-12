from general import *

X = np.random.rand(10,2)
X = X-0.5
theta = np.array([[5], [20]])
poly = np.matmul(X, theta)
hypo = sigmoid(poly)
calc_hypo(X, theta)

