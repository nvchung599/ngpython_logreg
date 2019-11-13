from general import *
from data_parser import *
from theta_optimizer import *

X, y = parse_data_ng('ex2data1.txt')
X_raw = np.copy(X)

#TODO INCORPORATEINCORPORATEINCORPORATEINCORPORATEINCORPORATEINCORPORATE

# get data from csv
#x, y = parse_data_ng('1.01. Simple linear regression.csv')

# construct feature matrix, normalize, bias
#X = mod_degree(x, 5)
X = normalize(X)
X = add_bias(X)
theta = construct_theta(X)
# train theta
my_opter = ThetaOptimizer(1000, 0.00000000001, 0.01)
theta, cost = my_opter.optimize_theta(X, y, 1)
#print(theta)
print(cost)
my_opter.plot_dec_bound(X_raw, y, theta)
#my_opter.plot_last()
#print(calc_hypo(X, theta))
