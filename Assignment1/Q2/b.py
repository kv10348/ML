from a import *

def compute_cost(X, Y, theta, m):
    A = np.subtract(np.dot(X, theta), Y)
    # print("---")
    # print(A)
    # print("---")
    C = 0.5 * (1 / m) * (np.dot(A.T, A))
    return C[0][0]


def SGD(X, Y, theta, alpha, epsilon, m, r, show, maxiterations, iteration, plot):
    itr = 0
    all_cost = []
    all_theta0 = []
    all_theta1 = []
    all_theta2 = []
    iterations = []
    print('batch-size ', r)
    while itr < maxiterations:
        itr += 1
        for start in range(0, m, r):
            X_ = X[start:min(r + start, m)]
            # print(X_)
            Y_ = Y[start:min(r + start, m)]
            # print(np.dot(X_, theta))
            # print(np.subtract(np.dot(X_, theta), Y_))
            # print(np.dot(X_.T,np.subtract(np.dot(X_, theta), Y_)))
            gradient = alpha *np.dot(X_.T,
                                      np.subtract(np.dot(X_, theta), Y_))/r
            # print(gradient)
            theta = theta - gradient
            # break
        cost = compute_cost(X, Y, theta, m)
        # print(cost)
        # break
        all_cost.append(cost)
        all_theta0.append(theta[0][0])
        all_theta1.append(theta[1][0])
        all_theta2.append(theta[2][0])
        n = len(all_cost)
        iterations.append(itr)
        # print(cost)
        if n > 1:
            if abs(all_cost[n - 1] - all_cost[n - 2]) > epsilon:
                pass
            else:
                print('The no. of iterations: {0}'.format(str(itr)))
                break
        if itr % iteration != 0 or not show:
            continue
        print(
            'The cost after {0} epoches for batch-size {1} is {2} and thetas are {3}, {4}, {5}'.format(str(itr), str(r),
                                                                                                       str(cost),
                                                                                                       str(theta[0][0]),
                                                                                                       str(theta[1][0]),
                                                                                                       str(theta[2][
                                                                                                               0])))
    if plot:
        plt.figure(2)
        plt.plot(iterations, all_cost, 'b-')
        plt.xlabel('<-- No of Epoches -->')
        plt.ylabel('<-- Cost of Model -->')
        plt.title('<-- Graph of Cost vs No of Epoches with alpha ' + str(float(alpha)) + ' -->')
        a = 'cost_shape_'+str(r)
        plt.savefig(a)
        plt.show()
    return theta,cost, all_cost, all_theta0, all_theta1, all_theta2


def main(X, Y, m, batches, alpha):
    epsilon = math.pow(10, -20)
    theta = np.reshape(np.zeros(3), (3, 1))
    # print('Cost for thetas:[0, 0, 0] :=' + str(compute_cost(X, Y, theta, m)))
    # Computing thetas using batch SGD

    all_thetas_ = []
    all_thetas = []
    maxiterations= 20000
    for i in range(0, len(batches)):
        t,cost, all_cost, all_theta0, all_theta1, all_theta2 = SGD(X, Y, theta, alpha, epsilon, m, batches[i], True, maxiterations,
                                                              100, True)
        #         print("Theta for batch size " + str(batches[i]) + " are ",  t[0][0], t[1][0], t[2][0])
        all_thetas.append([all_theta0, all_theta1, all_theta2])
        all_thetas_.append([t[0][0], t[1][0], t[2][0]])
        # print(all_cost)
        print('Thetas for batch size ' + str(batches[i]) + " are := {0}, {1}, {2} ".format(str(t[0]), str(t[1]), str(t[2])))
        print('The final cost: {0}'.format(str(cost)))



    return all_thetas, all_thetas_


batches = [1, 100, 10000, 1000000]

alpha = 0.001
# print(X[0:1].T)
all_thetas, theta_ = main(X, Y, m, batches, alpha)