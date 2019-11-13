from general import *
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
for i in range(nx):
    for j in range(ny):
        # treat xv[i,j], yv[i,j]

xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
for i in range(nx):
    for j in range(ny):
        # treat xv[j,i], yv[j,i]

X = np.random.rand(10,2)
X = X-0.5
theta = np.array([[5], [20]])
poly = np.matmul(X, theta)
hypo = sigmoid(poly)
calc_hypo(X, theta)

