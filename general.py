import numpy as np
from matplotlib import pyplot as plt

def split_data(data):
    """splits a data matrix 60/20/20. accepts arbitrary dataset sizes m and n."""
    m = np.size(data, 0)
    p1 = int(round(m*0.6))
    p2 = int(round(m*0.8))
    train = np.copy(data[:p1,:])
    test = np.copy(data[p1:p2,:])
    cv = np.copy(data[p2:,:])
    return train, cv, test

def init_nested_list(rows, cols):
    """initializes an empty (None) nested list/matrix. To be populated with
    vectors of varying lengths (theta vectors)
    note: to access... 2dlist[row][col]"""
    out_list = [None] * rows
    for i in range(0, rows):
        out_list[i] = [None] * cols #in_list
    return out_list

def split_into_nlist(my_list):
    """input a list of X matrices of increasing degrees.
    output a nested list of X matrices of increasing degrees, but split into
    training, cross-validation, and test sets. rows = degree. cols = set"""
    degrees = len(my_list)
    deg_set_data_nlist = init_nested_list(degrees, 3)
    for i in range(0, degrees):
        train, cv, test = split_data(my_list[i])
        deg_set_data_nlist[i][0] = train
        deg_set_data_nlist[i][1] = cv
        deg_set_data_nlist[i][2] = test
    return deg_set_data_nlist

def mod_degree(x, deg):
    """if degree is specified to be >1, creates additional features and
    constructs the matrix X"""
    X = np.copy(x)
    if deg > 1:
        for i in range(2, deg+1):
            add_me = np.copy(x)**i
            X = np.hstack([X, add_me])
    return X

def gen_degree_cases(x, max_deg):
    """returns a list of numpy matrices - X matrices of increasing degrees"""
    my_list = []
    for i in range(1, max_deg+1):
        my_list.append(mod_degree(x, i))
    return my_list


def construct_theta(X):
    """random inits a theta vector of compatible size with X.
    call this AFTER bias has been added to X"""
    n_plus_one = np.size(X,1)
    theta = np.random.rand(n_plus_one, 1)
    return theta

def normalize(X):
    """normalizes every feature column in the X matrix wrt mean and stdev
    note: apply this function BEFORE adding a ones column
    note: shape of matrix should be m*n"""
    n = np.size(X,1)
    X_norm = np.zeros(X.shape)
    for i in range(0, n):
        feature_vec = np.copy(X[:,[i]])
        my_std = np.std(feature_vec)
        my_mean = np.mean(feature_vec)
        feature_vec_norm = (feature_vec - my_mean)/my_std
        X_norm[:,[i]] = feature_vec_norm
    return X_norm

def get_stats(x):
    """normalizes every feature column in the X matrix wrt mean and stdev
    note: apply this function BEFORE adding a ones column
    note: shape of matrix should be m*n"""
    my_std = np.std(x)
    my_mean = np.mean(x)
    return my_std, my_mean

def normalize_list(my_list):
    """normalizes every element in a list of X matrices"""
    my_list_norm = []
    for X in my_list:
        my_list_norm.append(normalize(X))
    return my_list_norm

def add_bias(X):
    """Adds a column of ones on the left side of a 2D matrix
    note: a numpy matrix must be shape (m,1) NOT (m,)"""
    m = np.size(X,0)
    ones = np.ones([m,1])
    X_bias = np.hstack([ones, X])
    return X_bias

def sigmoid(x):
    """returns sigmoid of single value, or piece-wise sigmoid of matrix"""
    return (1/(1+np.exp(-x)))

def calc_hypo(X, theta):
    """X is an m*n matrix with bias added already"""
    poly = np.matmul(X, theta)
    hypo = sigmoid(poly)
    return hypo

def calc_cost(X, y, theta, reg_const):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = calc_hypo(X, theta) # (m*n)*(n*1) = m*1
    sum_me = y*np.log(hypo) + (1-y)*np.log(1-hypo)
    J_nonreg = (-1/m) * np.sum(sum_me)

    temp_theta = np.copy(theta)
    temp_theta[0] = 0
    temp_theta = temp_theta**2
    reg_term = (1/(2*m))*reg_const*np.sum(temp_theta)

    cost = J_nonreg + reg_term
    return cost

   # m = np.size(X,0)
   # hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
   # err = hypo - y
   # sqr_err = np.power(err, 2)
   # sum_sqr_err = np.sum(sqr_err)

   # temp_theta = np.copy(theta)
   # temp_theta[0] = 0
   # temp_theta = temp_theta**2
   # reg_term = reg_const*np.sum(temp_theta)

   # cost = (sum_sqr_err + reg_term)/(2*m)
   # return cost

def calc_grad(X, y, theta, reg_const):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
    err = hypo - y # m*1
    accum_term = np.matmul(np.transpose(X), err) # (4*m)*(m*1) = 4*1
    grad = (accum_term + reg_const*theta)/m
    return grad

def grad_check(X, y, theta, epsilon, reg_const):
    """numerically calculates parameter gradients for the first learning step.
    prints side by side with matrix-calculated gradients for verification"""
    n = np.size(theta,0)
    ep_mat = np.identity(n)*epsilon
    mat_grad = calc_grad(X, y, theta, reg_const)
    num_grad = np.zeros((n,1))
    for i in range(0,n):
        ep_vec = ep_mat[:,[i]]
        J_hi = calc_cost(X, y, theta+ep_vec, reg_const)
        J_lo = calc_cost(X, y, theta-ep_vec, reg_const)
        num_grad[i,0] = (J_hi-J_lo)/(2*epsilon)
    print(mat_grad)
    print(num_grad)

def map_scatter(list_x, list_y, nlist):
    """maps python lists and nested list into 3 numpy vectors for scatter plotting"""
    el_qty = len(list_x)*len(list_y) 
    x = np.zeros(el_qty)
    y = np.zeros(el_qty)
    z = np.zeros(el_qty)
    k = 0
    for i in range(len(list_x)):
        for j in range(len(list_y)):
            x[k] = list_x[i]
            y[k] = list_y[j]
            z[k] = nlist[i][j]
            k = k + 1
    return x, y, z


#def calc_hypo(x_plotvec, theta_opt):
#    """given raw x 1d vector, mod the degree to be consistent with the
#    theta_opt vector, normalize, add bias, then predict."""
#    X = mod_degree(x_plotvec, np.size(theta_opt, 0) - 1)
#    X = normalize(X)
#    X = add_bias(X)
#    y = np.matmul(X, theta_opt)
#    return y


