import math
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def Sampling_data(mean1, sigma1, mean2, sigma2, m, theta):
    x1 = np.array([np.random.normal(mean1, sigma1, m)]).astype(np.float).T
    x2 = np.array([np.random.normal(mean2, sigma2, m)]).astype(np.float).T
    x3 = np.hstack((np.reshape(np.ones(m), (m, 1)), x1))
    X = np.hstack((x3, x2))
    epsilon = np.array([np.random.normal(0, 1.414, int(1e6))]).astype(np.float).T
    Y = np.array(np.dot(X, theta)).astype(np.float)+epsilon
    return X, Y


theta = [3, 1, 2]
theta = np.reshape(theta, (3, 1))
m = int(1e6)
X, Y = Sampling_data(3, 2, -1, 2, m, theta)
