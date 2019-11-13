from general import *
from data_parser import *

X, y = parse_data_ng('ex2data1.txt')

pos_i = y==1
pos_i = pos_i.reshape(-1)
neg_i = y==0
neg_i = neg_i.reshape(-1)

X_pos = X[pos_i, :]
y_pos = y[pos_i, :]
X_neg = X[neg_i, :]
y_neg = y[neg_i, :]

plt.title('Data Visualization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X_pos[:,0], X_pos[:,1], marker='+', label='positive')
plt.scatter(X_neg[:,0], X_neg[:,1], marker='o', label='negative')
plt.legend()
plt.show()

