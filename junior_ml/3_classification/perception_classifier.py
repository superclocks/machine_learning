import numpy as np
import random
import math
import matplotlib.pyplot as plt
from synthic_data import twoSamples
def perSGD(x, y):
    w0 = np.array([0.2, 0.2])
    b0 = 0.1
    lamda = 0.1
    while 1:
        k = 0
        for i in xrange(len(x)):
            xi = x[i]
            yi = y[i]
            if yi * (w0[0] * xi[0] + w0[1] * xi[1] + b0) <= 0.0:
                w0 = w0 + lamda * yi * w0
                b0 = b0 + lamda * yi
                k = k + 1
        if k == 0:
            break
    return w0, b0
def perBatch(x, y):
    w0 = np.array([-0.1, -0.1])
    b0 = -0.1
    lamda = 0.01
    k = 0
    while k < 1000:
        g_w = [0.0, 0.0]
        g_b = 0.0
        for i in xrange(len(x)):
            xi = x[i]
            yi = y[i]
            if yi * (w0[0] * xi[0] + w0[1] * xi[1] + b0) <= 0.0:
                g_w = g_w + yi * w0
                g_b = g_b + yi
        if abs(g_b) <= 1e-16 and (abs(g_w[0]) + abs(g_w[1]) < 1e-16):
            break
        w0 = w0 +lamda * g_w
        b0 = b0 + lamda * g_b 
        k = k + 1
    return w0, b0
def plotDL(x, y, w, b):
    px = [0.0, -b/w[1]]
    py = [-b/w[0], 0.0]
    x = np.array(x)
    y = np.array(y)
    plt.plot(x[y == 1, :][:,0], x[y == 1, :][:,1], 'ro', markersize = 12)
    plt.plot(x[y == -1, :][:,0], x[y == -1, :][:,1], 'go', markersize = 12)

    plt.plot(px, py)
    plt.legend(('positive','negtive'))
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('perception classifier')
    plt.show()
    
if __name__ == '__main__':
    [x, y] = twoSamples()
    w0, b0 = perSGD(x, y)
    plotDL(x, y, w0, b0)