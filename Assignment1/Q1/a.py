import math
import numpy as np
import csv
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def compute_cost(X, Y, theta, m):
    A = np.subtract(np.dot(X, theta), Y)
    # print("---")
    # print(A)
    # print("---")
    C = 0.5 * (1 / m) * (np.dot(A.T, A))
    return C[0][0]


def Load_data(x_path, y_path):
    x_training = []
    y_training = []
    with open(x_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            x_training.append(row[0])
    with open(y_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            y_training.append(row[0])
    return x_training, y_training


def normalize_training_data(x_training):  # Normalizing the training data
    x_training = (x_training - np.mean(x_training)) / (np.std(x_training))
    # print(x_training)
    return x_training


def gradient_descent(X, Y, theta, alpha, epsilon, m, show, maxiterations, iteration, plot, name):
    i = 0
    all_cost = []
    all_theta0 = []
    all_theta1 = []
    itr = []
    converge = 1
    while i < maxiterations:
        # print(i)
        i += 1
        gradient = alpha * np.dot(X.T, np.subtract(np.dot(X, theta), Y))/m
        cost = compute_cost(X, Y, theta, m)
        all_cost.append(cost)
        itr.append(i)
        all_theta0.append(theta[0])
        all_theta1.append(theta[1])
        n = len(all_cost)
        theta = theta - gradient
        if n > 1:
            # print(abs(all_cost[n - 1] - all_cost[n - 2]))
            if abs(all_cost[n - 1] - all_cost[n - 2]) > epsilon :

                pass
            else:
                print('No of iterations: {0}'.format(str(i)))
                break
            if all_cost[n-1]-all_cost[n-2] < pow(10, 3):
                pass
            else:
                converge = 0
                break
        if i % iteration != 0 or not show:
            continue
        print('The cost after ' + str(i) + ' iterations := ' + str(cost) + ' and thetas are ' + str(theta[0][0])+ ', '+ str(theta[1][0]))
    if plot:
        plt.figure(2)
        plt.plot(itr, all_cost, 'b-')
        plt.xlabel('<-- No of iterations -->')
        plt.ylabel('<-- Cost of Model -->')
        plt.title('<-- Graph of Cost vs No of iteration with alpha ' + str(float(alpha)) + ' -->')
        plt.savefig(name)
        plt.show()
        # input('Hit enter to close')
        # plt.close()
    return theta, cost, all_theta0, all_theta1, all_cost, converge


def main(x_training, y_training, alpha, m, name):
    x_training = x_training.T
    Y = y_training.T

    t_ = normalize_training_data(x_training)
    X = np.hstack((np.reshape(np.ones(m), (m, 1)), t_))

    theta = np.reshape(np.zeros(2), (2, 1))

    # Setting Learning parameters

    epsilon = math.pow(10, -20)


    maxepoches = 20000
    # Computing theta using Gradient Descent algorithms

    theta, cost, thetas0, thetas1, all_cost, c = gradient_descent(X, Y, theta, alpha, epsilon, m, 1, maxepoches, 100, True, name)
    print('Final values for theta0 and theta1 are:= {0}, {1}'.format(str(theta[0]),
                                                                     str(theta[1])))
    print('The learning rate: {0}'.format(str(alpha)))
    print('Stopping criteria: Max epoches {0}; precision {1}'.format(str(maxepoches), str(epsilon)))
    print('Cost with thetas:[0,0] :=' + str(compute_cost(X, Y, theta, m)))
    print('The final cost is:= ' + str(cost))
    if c==0:
        print('The function is diverging.')



    return X, Y, x_training, y_training, theta, thetas0, thetas1


x_, y_ = Load_data("linearX.csv", "linearY.csv")
m = len(y_)
# print(m)
alpha= 0.015
name = 'cost_'+str(alpha)+'_.png'
X, Y, x_training, y_training, theta, thetas0, thetas1 = main(np.array([x_]).astype(np.float), np.array([y_]).astype(np.float), alpha, m, name )
