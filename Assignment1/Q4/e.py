from a import *


def plot_training_set(X, Y):
    m = len(Y)
    plt.figure(1)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    for i in range(0, m):
        if Y[i][0] == 1:
            y1, = plt.plot(X[i, 0], X[i, 1], 'ro', markersize=4)
        else:
            y2, = plt.plot(X[i, 0], X[i, 1], 'b^', markersize=4)
    plt.legend(handles=[y1, y2], labels=['Alaska', 'Canada'], loc='upper left', fontsize=8)


# part e
plot_training_set(X, Y)
plt.title('Boundary decision using linear and quadratic GDA')
decision_boundary(mu_0, mu_1, sigma, sigma, 'b', p)
decision_boundary(mu_0, mu_1, sigma_0, sigma_1, 'g', p)
plt.savefig('trained_data_linear_and_quadratic_GDA ')
plt.show()
