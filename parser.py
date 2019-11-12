import numpy as np


def hello_world():
    print("hello world")

def parse_data_ng(path):
    """extracts csv data and cleans it up
    input data must be 2D (x and y)
    output must be and m*1 numpy matrix/vector, homogenous, of same shape/size"""
    data = np.genfromtxt(path, delimiter=',')
    #np.random.shuffle(data)
    X = data[:,0:2]
    y = data[:,2]
    y = y.reshape(-1,1)
    return (X,y)
