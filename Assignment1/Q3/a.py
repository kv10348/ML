import math
import numpy as np
import csv
from matplotlib import cm
import matplotlib.pyplot as plt


def Load_data(x_path, y_path):
    x_training = []
    y_training = []
    with open(x_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            x_training.append([row[0], row[1]])
    with open(y_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            y_training.append([row[0]])
    return np.array(x_training).astype(np.float), np.array(y_training).astype(np.float)


def normalize_training_data(x_training):  # Normalizing the training data
    x_training[:, 0] = (x_training[:, 0] - np.mean(x_training[:, 0])) / (np.std(x_training[:, 0]))
    x_training[:, 1] = (x_training[:, 1] - np.mean(x_training[:, 1])) / (np.std(x_training[:, 1]))

    # print(x_training)
    return x_training

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_cost(X, Y, theta):
    ans = 0
    for i in range(0, len(Y)):
        if Y[i][0] == 1:
            ans += np.sum(np.log(sigmoid(np.dot(X[i], theta)[0])))
        else:
            ans += np.sum(np.log(1 - sigmoid(np.dot(X[i], theta)[0])))
    return ans


def compute_hessian(X, Y, theta):
    m = len(Y)
    H = np.reshape(np.zeros((m, m)), (m, m))
    temp = sigmoid(np.dot(X, theta))
    for i in range(0, m):
        H[i, i] = temp[i] * (1 - temp[i])
    return np.dot(X.T, np.dot(H, X))


def newton_method(X, Y, theta, epsilon, show, maxiterations, iteration, plot):
    all_cost = []
    all_theta0 = []
    all_theta1 = []
    all_theta2 = []
    itr = 0
    iterations = []
    while itr < maxiterations:
        itr += 1
        cost = compute_cost(X, Y, theta)
        all_cost.append(cost)
        iterations.append(itr)
        gradient = np.dot(X.T, sigmoid(np.dot(X, theta))-Y)
        H = compute_hessian(X, Y, theta)
        theta -= np.dot(np.linalg.inv(H),gradient)
        all_theta0.append(theta[0])
        all_theta1.append(theta[1])
        all_theta2.append(theta[2])

        n = len(all_cost)
        if n > 1:
            if abs(all_cost[n-1]-all_cost[n-2])> epsilon:
                pass
            else:
                break
        if itr % iteration != 0 or not show:
            continue
        print('Cost after {0} epoches is and thetas are {2}, {3}, {4} '.format(str(cost), str(theta[0]), str(theta[1]), str(theta[2])))

    if plot:
        plt.figure(2)
        plt.plot(iterations, all_cost, 'b-')
        plt.xlabel('<-- No of Epoches -->')
        plt.ylabel('<-- Cost of Model -->')
        plt.title('<-- Graph of Cost vs No of Epoches -->')
        plt.savefig('cost_shape')
        plt.show()
    return theta, all_cost, all_theta0, all_theta1, all_theta2

def main(x_training, y_training):
    m = len(y_training)
    # print(m)
    X = np.hstack((np.reshape(np.ones(m), (m, 1)), normalize_training_data(x_training)))
    Y = y_training

    theta = np.reshape(np.zeros(3), (3, 1))
    compute_hessian(X, Y, theta)
    # print(np.dot(X,
    # print(Y)
    # print(X)
    alpha = 0.0015
    maxiterations = 20000
    theta, all_cost, all_theta0, all_theta1, all_theta2 = newton_method(X, Y, theta, pow(10, -20), True, maxiterations, 100, True)
    print('The final cost value is: {0}  '.format(str(all_cost[len(all_cost)-1])))
    print('The final thetas values are {0}, {1}, {2}'.format(str(theta[0][0]),str(theta[1][0]),str(theta[2][0])))
    return X, Y, theta

x_training, y_training = Load_data("logisticX.csv", "logisticY.csv")
x_training = normalize_training_data(x_training)
X, Y, theta = main(x_training, y_training)