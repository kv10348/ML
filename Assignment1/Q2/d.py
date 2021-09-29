from b import *




fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Theta2')
color = ['y', 'b', 'k', 'r']
j = 0
for i in all_thetas:
    if j == 0:
        print("Yellow color corresponds to batch size of 1")
    if j == 1:
        print("Blue color corresponds to batch size of 100")
    if j == 2:
        print("Black color corresponds to batch size of 10000")
    if j == 3:
        print("red color corresponds to batch size of 1000000")
    for a in range(0, len(i[0])):
        ax.scatter(i[0][a], i[1][a], i[2][a], c=color[j], marker='o', s=2)
    j += 1
plt.savefig('thetas_shapes')
plt.show()