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


# part b
plt.title('Training_data')
plot_training_set(X, Y)
plt.savefig('training_data')
plt.show()