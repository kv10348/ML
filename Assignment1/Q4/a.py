import math
import numpy as np
import csv
from matplotlib import cm
import matplotlib.pyplot as plt




# Alaska : 1
# Canada : 0

def Load_data(x_path, y_path):
    x_training = []
    y_training = []
    with open(x_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=' ')
        for row in read_csv:
            x_training.append([float(row[0]), float(row[2])])
    with open(y_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            if (row[0] == 'Alaska'):
                y_training.append([1])
            else:
                y_training.append([0])
    return np.array(x_training).astype(np.float), np.array(y_training).astype(np.float)


def normalize_training_data(x_training):  # Normalizing the training data
    x_training[:, 0] = (x_training[:, 0] - np.mean(x_training[:, 0])) / (np.std(x_training[:, 0]))
    x_training[:, 1] = (x_training[:, 1] - np.mean(x_training[:, 1])) / (np.std(x_training[:, 1]))

    # print(x_training)
    return x_training

def mu_sigma(X, Y):
    m = len(Y)
    mu_0 = np.reshape(np.zeros(2), (2, 1))
    mu_1 = np.reshape(np.zeros(2), (2, 1))
    sigma = np.reshape(np.zeros((2, 2)), (2, 2))
    sigma_0 = np.reshape(np.zeros((2, 2)), (2, 2))
    sigma_1 = np.reshape(np.zeros((2, 2)), (2, 2))
    a = 0
    b = 0
    for i in range(m):
        if Y[i][0] == 1:
            a += 1
            mu_1 += np.reshape(X[i], (2, 1))
        else:
            b += 1
            mu_0 += np.reshape(X[i], (2, 1))

    mu_0 = mu_0 / b
    mu_1 = mu_1 / a
    for i in range(m):
        if Y[i][0] == 1:

            sigma_ = np.reshape(X[i], (2, 1)) - mu_1
            temp = np.dot(sigma_, sigma_.T)
            sigma_1 += temp
            sigma += temp
        else:

            sigma_ = np.reshape(X[i], (2, 1)) - mu_0
            temp = np.dot(sigma_, sigma_.T)
            sigma_0 += temp
            sigma += temp

    sigma = sigma / m
    sigma_1 = sigma_1 / a
    sigma_0 = sigma_0 / b

    return mu_0, mu_1, sigma_0, sigma_1, sigma, a / m


def decision_boundary_helper(mu_0, mu_1, sigma_0, sigma_1, p, x):
    a = np.array(2 * np.log(p / (1 - p))) + np.log(np.linalg.det(sigma_0) / np.linalg.det(sigma_1))
    b = np.dot(np.dot((x - mu_1).T, np.linalg.inv(sigma_1)), (x - mu_1)) - np.dot(
        np.dot((x - mu_0).T, np.linalg.inv(sigma_0)), (x - mu_0))
    return a - b


def decision_boundary(mu_0, mu_1, sigma_0, sigma_1, c, p):
    x_1 = np.linspace(-2.5, 2.5, 50)
    x_2 = np.linspace(-2.5, 2.5, 50)
    x_1, x_2 = np.meshgrid(x_1, x_2)

    m = np.zeros(np.shape(x_1))
    for i in range(len(m)):
        for j in range(len(m[0])):
            z = np.array([[x_1[i][j]], [x_2[i][j]]]).astype(np.float)

            m[i][j] = decision_boundary_helper(mu_0, mu_1, sigma_0, sigma_1, p, z)

    plt.contour(x_1, x_2, m, levels=[0], colors=c)

X, Y = Load_data('q4x.dat', 'q4y.dat')
X = normalize_training_data(X)
m = len(Y)
mu_0, mu_1, sigma_1, sigma_0, sigma, p = mu_sigma(X, Y)
# part a
print('The mu_0 : [{0}, {1}]'.format(str(mu_0[0][0]), str(mu_0[1][0])))
print('The mu_1 : [{0}, {1}]'.format(str(mu_1[0][0]), str(mu_1[1][0])))
print('The sigma: ', str(sigma))






