from a import *




def plot_training_set(x_training, y_training):
    print('Plotting the training set')
    plt.figure(1)
    plt.plot(x_training, y_training, 'r+')
    plt.xlabel("<-- Acidity -->")
    plt.ylabel("<-- Density -->")
    plt.title('Training Set')
    plt.savefig("training_data")
    plt.show(block=True)
    # input('\nPress Enter to close\n')


def plot_trained_model(X, x_training, Y, theta):
    plt.figure(3)
    print('Plotting the trained data set')
    plt.plot(x_training, Y, 'r+', label='Input Example', markersize=5)
    plt.xlabel('Acidity -->')
    plt.ylabel('Density -->')
    plt.title('Prediction Model using Linear Regression')
    plt.plot(x_training, np.dot(X, theta), 'g-', label='Prediction Line')
    plt.legend()
    plt.savefig("trained_data")
    plt.show()
    # input('\nPress Enter to close\n')
    # plt.close()
plot_training_set(x_training, Y)
plot_trained_model(X, x_training, Y, theta)