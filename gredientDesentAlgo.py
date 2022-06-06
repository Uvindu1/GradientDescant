import numpy as np
import matplotlib.pyplot as plt

# number of data point
n=5

#simple data set
x = np.array([1, 2, 3, 4, 5])
y = np.array([5,8,11,14,17])

# start with m = 0, c = 0
m = 0
c = 0
learning_rate = 0.01  # alfa agaya


# algorithem eka
for i in range(1, 101):
    y_predict = m * x +c
    # cost eka calcuate karano
    cost = (1/n) * sum([value ** 2 for value in (y - y_predict)])

    #plot after each iteration cost against m
    plt.scatter(m, cost)

    # calculate gradients (awakalanaya bavithaya)
    dm = -(2/n)* sum(x * (y - y_predict))
    dc = -(2/n)* sum(y-y_predict)

    # update parameters (kuda agayakin wenas karana piyawara)
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    print(f'm:{m} c:{c}')
plt.show()




































