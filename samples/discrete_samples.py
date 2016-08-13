import random
import numpy as np
import math
import matplotlib.pyplot as plt

def demo(n, p):
    U = np.zeros(n)
    for i in xrange(n):
        U[i] = random.random()
    X = np.zeros(n)
    w1 = U <= p[0]
    X[w1] = 1
    
    w2 = (U > p[0]) & (U < sum(p[0:2]))
    X[w2] = 2
    
    w3 = U > sum(p[0:2])
    X[w3] = 3
    return X

def posiCDF(x, L):
    r = []
    for i in xrange(0, x+1):
        r.append(np.exp(-L)*np.power(L,i)/float(math.factorial(i)))
    return r
def poisSamples(n, L):
    U = np.zeros(n)
    for i in xrange(n):
        U[i] = random.random()
    X = np.zeros(n)
    for i in xrange(1, n):
        if(U[i] < posiCDF(0, L)):
            X[i] = 0
        else:
            B = False
            I = 0
            while(B == False):
                r1 = posiCDF(I, L)
                r2 = posiCDF(I + 1, L)
                inte = [sum(r1),sum(r2)]
                
                if((U[i] > inte[0]) and (U[i] < inte[1])):
                    X[i] = I + 1
                    B = True
                else:
                    I = I + 1
    return X
            
def posiDemo():
    V = poisSamples(10000, 4)
    n = 15
    MF = np.zeros(n)
    for i in xrange(n):
        MF[i] = np.mean(V == float(i))
    
    b = np.arange(0, n)
    plt.plot(b, MF)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    
    z = posiCDF(15, 4)
    plt.plot(z,'r')
    plt.legend(('sampes','true'))
    plt.show()
if __name__ == '__main__':
    posiDemo()
    X = demo(10000, [0.4, 0.25, 0.35])
    print np.mean(X == 1)
    print np.mean(X == 2)
    print np.mean(X == 3)



    