import d
import a
import numpy as np


alphas = [0.001, 0.025, 0.1]
x_, y_ = a.Load_data('linearX.csv', 'linearY.csv')
m = len(y_)

for alpha in alphas:
    print('*********************************')
    name = 'cost_'+str(alpha)+'_.png'
    X, Y, x_training, y_training, theta, thetas0, thetas1 = a.main(np.array([x_]).astype(np.float),
                                                                 np.array([y_]).astype(np.float), alpha, m, name)
    name = 'contour_part_e_'+str(alpha)+'.png'
    d.plot_contours(X, Y, thetas0, thetas1, alpha, name)
