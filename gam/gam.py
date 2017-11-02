'''original example for checking how far GAM works

Note: uncomment plt.show() to display graphs
'''

example = 1  # 1,2 or 3
from math import exp, pow
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM

standardize = lambda x: (x - x.mean()) / x.std()
demean = lambda x: (x - x.mean())
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
y = R.standard_normal((nobs,))

f1 = lambda x1: (x1 + x1**2 - 3 - 1 * x1**3 + 0.1 * np.exp(-x1/4.))
f2 = lambda x2: (x2 + x2**2 - 0.1 * np.exp(x2/4.))
z = standardize(f1(x1)) + standardize(f2(x2))
z = standardize(z) * 2 # 0.1

y += z
d = np.array([x1,x2]).T
def sampleFun(x, y):
    z =  3*pow((1-x),2) * exp(-(pow(x,2)) - pow((y+1),2)) \
   - 10*(x/5 - pow(x, 3) - pow(y, 5)) * exp(-pow(x, 2) - pow(y, 2)) \
   - 1/3*exp(-pow((x+1), 2) - pow(y, 2)) 
    return z
def peaksSamples(n):
    x = np.array([np.linspace(-3, 3, n)])
    x = x.repeat(n, axis = 0)
    y = x.transpose()
    z = np.zeros((n, n))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i][j] = sampleFun(x[i][j], y[i][j])
    X = np.zeros((n*n, 2))
    x_list = x.reshape(n*n,1 )
    y_list = y.reshape(n*n,1)
    z_list = z.reshape(n*n,1)
    n = 0
    for xi, yi in zip(x_list, y_list):
        X[n][0] = xi
        X[n][1] = yi
        n = n + 1
    
    return X,z_list.transpose()

if example == 1:
    f = famil

    print "normal"
    [X, z] = peaksSamples(100)
    m = AdditiveModel(X,family=f)
    zz = z.transpose()
    m.fit(z[0])
    x = np.linspace(-2,2,50)

    print m

    y_pred = m.results.predict(X)
    plt.figure()
    plt.plot(y, '.')
    plt.plot(z.transpose(), 'b-', label='true')
    plt.plot(y_pred, 'r-', label='AdditiveModel')
    plt.legend()
    plt.title('gam.AdditiveModel')

import scipy.stats, time

if example == 2:
    print "binomial"
    f = family.Binomial()
    b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(y)])
    b.shape = y.shape
    m = GAM(b, d, family=f)
    toc = time.time()
    m.fit(b)
    tic = time.time()
    print tic-toc


if example == 3:
    print "Poisson"
    f = family.Poisson()
    y = y/y.max() * 3
    yp = f.link.inverse(y)
    p = np.asarray([scipy.stats.poisson.rvs(p) for p in f.link.inverse(y)], float)
    p.shape = y.shape
    m = GAM(p, d, family=f)
    toc = time.time()
    m.fit(p)
    tic = time.time()
    print tic-toc


plt.figure()
plt.plot(x1, standardize(m.smoothers[0](x1)), 'r')
plt.plot(x1, standardize(f1(x1)), linewidth=2)
plt.figure()
plt.plot(x2, standardize(m.smoothers[1](x2)), 'r')
plt.plot(x2, standardize(f2(x2)), linewidth=2)




plt.show()



##     pylab.figure(num=1)
##     pylab.plot(x1, standardize(m.smoothers[0](x1)), 'b')
##     pylab.plot(x1, standardize(f1(x1)), linewidth=2)
##     pylab.figure(num=2)
##     pylab.plot(x2, standardize(m.smoothers[1](x2)), 'b')
##     pylab.plot(x2, standardize(f2(x2)), linewidth=2)
##     pylab.show()

