from general import *
from parser import *
from theta_optimizer import *

hello_world()
X, y = parse_data_ng('ex2data1.txt')


#TODO INCORPORATEINCORPORATEINCORPORATEINCORPORATEINCORPORATEINCORPORATE

# get data from csv
#x, y = parse_data_ng('1.01. Simple linear regression.csv')

# construct feature matrix, normalize, bias
#X = mod_degree(x, 5)
X = normalize(X)
X = add_bias(X)
# train theta
my_opter = ThetaOptimizer(100, 0.000001, 0.1)
theta, cost = my_opter.optimize_theta(X, y, 1)
my_opter.plot_last()
