from math import pi,sqrt
import numpy as np
import random
import matplotlib.pyplot as plt
def pnorm(n, mean = 0, sd = 1):
    r = []
    if(type(mean) == np.ndarray):
        for i in xrange(n):
            r.append(random.normalvariate(mean[i], sd))
    else:
        for i in xrange(n):
            r.append(random.normalvariate(mean, sd))
    return r
def rbvn(n, rho):
    x = pnorm(n, 0.0, 1.0)
    y = pnorm(n, np.dot(rho, x), sqrt(1 - rho * rho))
    return [x, y]
def plotRbvn(r, k = 0):
    x = r[0]
    y = r[1]
    fig = plt.figure(k)
    ax = fig.add_subplot(321)
    ax.plot(x, y, '.')
    plt.xlabel('x')
    ax = fig.add_subplot(322)
    ax.plot(x, y)
    plt.ylabel('y')
    
    ax = fig.add_subplot(323)
    ax.plot(x)
    plt.title('x')
    ax = fig.add_subplot(324)
    ax.plot(y)
    plt.title('y')
    
    ax = fig.add_subplot(325)
    ax.hist(x, 50)
    plt.title('x distribution')
    ax = fig.add_subplot(326)
    ax.hist(y, 50)
    plt.title('y distribution')
    plt.show()
def gibbs(n, rho):
    x_list = []
    y_list = []
    x = [0.0]
    y = [0.0]
    for i in xrange(1, n):
        x = pnorm(1, rho * y[0], sqrt(1.0 - rho * rho))
        y = pnorm(1, rho * x[0], sqrt(1.0 - rho * rho))
        x_list.append(x[0])
        y_list.append(y[0])
    return [x_list, y_list]
if __name__ == '__main__':
    rho = 0.98
    #r = rbvn(10000, rho)
    #plotRbvn(r)
    r = gibbs(10000, rho)
    plotRbvn(r, k = 1)