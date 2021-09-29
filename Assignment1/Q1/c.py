from a import *


def plot_3d_mesh(X,Y,theta, learning_rate):
    x_theta = np.linspace(-0.2,2.0,num=100)
    y_theta = np.linspace(-1,1,num=100)

    x1, y1 = np.meshgrid(x_theta,y_theta)
    z1 = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            z1[i][j] = compute_cost(X,Y,np.array([[x1[i][j]], [y1[i][j]]]), m)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(15, 200)
    ax.set_title('path taken by GD in 3-D')

    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2')
    ax.set_zlabel('cost')

    ax.plot_surface(x1,y1,z1,rstride=1, cstride=1, cmap='viridis', edgecolor='none',antialiased=True, alpha = 0.5)
    for i in range(len(thetas0)):
        ax.scatter(thetas0[i], thetas1[i], compute_cost(X,Y,np.array([ thetas0[i], thetas1[i]] ) , m), s = 2, color = "black" )
        plt.pause(0.2)
    plt.savefig('3d_mesh_plot')
    plt.show()

plot_3d_mesh(X, Y, theta, 0.001)