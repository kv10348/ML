from a import *


def plot_training_set(X, Y):

    m = len(Y)
    plt.figure(1)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    for i in range(0, m):
        if Y[i][0] == 1:
            y1, = plt.plot(X[i, 1], X[i, 2], 'ro', markersize=4)
        else:
            y2, = plt.plot(X[i, 1], X[i, 2], 'b^', markersize=4)
    plt.legend(handles=[y1, y2], labels=['Y=1', 'Y=0'], loc='upper left', fontsize=8)
    plt.savefig('training_data')
    plt.show()


def plot_trained_data(X, Y, theta):
    m = len(Y)
    plt.figure(3)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    for i in range(0, m):
        if Y[i][0] == 1:
            y1, = plt.plot(X[i,1], X[i, 2], 'ro', markersize= 4)
        else:
            y2, = plt.plot(X[i,1], X[i,2], 'b^', markersize = 4)
    x_lines = [np.min(X[:, 1]), np.max(X[:, 1])]
    db, = plt.plot(x_lines, [(-(theta[0][0] + theta[1][0] * x_lines[0]) / theta[2][0]), (-(theta[0][0] + theta[1][0] * x_lines[1]) / theta[2][0])], color='k')
    plt.legend(handles=[y1, y2, db], labels=['Y=1', 'Y=0', 'Decision Boundary'], loc='upper left', fontsize=8)
    plt.savefig('trained_model')
    plt.show()

plot_training_set(X, Y)
plot_trained_data(X, Y, theta)