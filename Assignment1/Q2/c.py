from b import *


def Load_data(path):
    x_training = []
    y_training = []
    with open(path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            x_training.append([row[0], row[1]])
            y_training.append([row[2]])
    x_training.pop(0)
    y_training.pop(0)
    return np.array(x_training).astype(np.float), np.array(y_training).astype(np.float)


X1, Y1 = Load_data("q2test.csv")
m = len(Y1)
X1 = np.hstack((np.reshape(np.ones(m), (m, 1)), X1))
print("The cost with the original hypothesis : {0}".format(str(compute_cost(X1, Y1, theta, m))))
for i in range(0, len(batches)):
    print("The cost with the learning parameters from the model with batch-size {0}: {1}".format(str(batches[i]),
                                                                                                 str(compute_cost(X1,
                                                                                                                  Y1,
                                                                                                                  np.reshape(
                                                                                                                      theta_[
                                                                                                                          i],
                                                                                                                      (
                                                                                                                      3,
                                                                                                                      1)).astype(
                                                                                                                      np.float),
                                                                                                                  m))))

