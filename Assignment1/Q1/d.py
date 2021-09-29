

from a import *

def plot_contours(X,Y,thetas0, thetas1, learning_rate, name):
    x_theta = np.linspace(-0.2, 2.0, 20)
    y_theta = np.linspace(-1, 1, 20)

    x1, y1 = np.meshgrid(x_theta,y_theta)
    z1 = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            z1[i][j]= compute_cost(X, Y, np.array([[x1[i][j]],[y1[i][j]] ]), m)
    # zs = np.array([
    #     compute_cost(X, Y, np.array([[a_], [b_]]), m)
    #     for a_, b_ in zip(np.ravel(x1), np.ravel(y1))
    # ])
    # z1 = zs.reshape(x1.shape)

    plt.title('Contours of Error')
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')

    d = plt.contour(x1, y1, z1, levels=30)
    # plt.clabel(d, inline=1, fontsize=10)
    for i in range(len(thetas0)):
        plt.scatter(thetas0[i], thetas1[i], s = 1, c = "red")
        plt.pause(0.2)

    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    name = 'contours_'+str(0.015)+'_.png'
    plot_contours(X, Y, thetas0, thetas1, 0.015, name)