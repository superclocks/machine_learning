import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from synthic_data import linearSamples, nlSamples
from copy import deepcopy
from scipy import linalg

#closed-form for linear regression
def lR(x, y):
    x = np.matrix(x)
    if x.shape[0] == 1:
        x = x.transpose()
    y = np.matrix(y)
    if y.shape[0] == 1:
        y = y.transpose()
    one = np.ones((x.shape[0], 1))
    x = np.hstack([one, x])
    w = inv((x.transpose()).dot(x)).dot(np.transpose(x)).dot(y)
    return w
def plotLM(w, x,y):
    xx = [i for i in np.arange(0.0,20.0,0.5)]
    yy = [w[0,0] + w[1,0] * i for i in xx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'ro')
    ax.plot(xx,yy)
    s = 'y = %s + %s * x' %(str(w[0,0])[0:7], str(w[1, 0])[0:7])
    ax.annotate(s, xy=(xx[20], yy[20]),  xycoords='data',
                xytext=(-180, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(('training sampes','regression line'))
    plt.show()
#gradient descent for linear regression 
def obj(x, y, w):
    t = x.dot(w) - y
    t = np.multiply(t, t)
    sum_ = 0.5 * np.sum(t)
    return sum_
def gradient(fun, x, y, w, delta = 1e-6, *args):
    l = len(w)
    g = []
    for i in range(0, l):
        delta_w = deepcopy(w)
        delta_w[i] = delta_w[i] + delta
        g.append(-(obj(x, y, delta_w) - obj(x, y, w))/delta)
    return g
    
def gdLR(fun, x, y, w = [0.0, 0.0], step = 0.0007,tol = 1e-6):
    #preprocess the data
    x = np.matrix(x)
    if x.shape[0] == 1:
        x = x.transpose()
    y = np.matrix(y)
    if y.shape[0] == 1:
        y = y.transpose()
    one = np.ones((x.shape[0], 1))
    x = np.hstack([one, x])
    w = np.matrix(w)
    if w.shape[0] == 1:
        w = w.transpose()
    l = len(w)
    k = 1
    while(True):
        step1 = step / k 
        #1)compute negative gradient
        g = gradient(fun, x, y, w)
        err = linalg.norm(g)
        print err
        if err < tol or k > 200:
            break
        #2)updata the parameters
        w = [w[i,0] + step * g[i] for i in range(0, l)]
        w = np.matrix(w).transpose()
        k = k + 1
    return w
def plotGdLM(cf_w,gd_w, x,y):
    xx = [i for i in np.arange(0.0,20.0,0.5)]
    cf_yy = [cf_w[0,0] + cf_w[1,0] * i for i in xx]
    gd_yy = [gd_w[0,0] + gd_w[1,0] * i for i in xx]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, '.')
    ax.plot(xx,cf_yy,color = 'g', linewidth=3)
    s = 'y = %s + %s * x' %(str(cf_w[0,0])[0:7], str(cf_w[1, 0])[0:7])
    ax.annotate(s, xy=(xx[int(len(xx)/2)], cf_yy[int(len(xx)/2)]),  xycoords='data',
                xytext=(-180, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc='g', ec='g'),
                arrowprops=dict(arrowstyle="->",fc='g', ec='g',
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    ax.plot(xx,gd_yy, color = 'r', linewidth=3)
    s = 'y = %s + %s * x' %(str(gd_w[0,0])[0:7], str(gd_w[1, 0])[0:7])
    ax.annotate(s, xy=(xx[int(len(xx)/2)+5], cf_yy[int(len(xx)/2)+5]),  xycoords='data',
                xytext=(-180, 30), textcoords='offset points',
                bbox=dict(boxstyle="round",  fc='r', ec='r'),
                arrowprops=dict(arrowstyle="->", fc='r', ec='r',
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(('training sampes', 'closed-form regression','gradient descent regression'),loc='upper center')
    plt.show()        
    
#basis function for linear regression
def lrForNLSamples():
    [x, y] = nlSamples()
    #plt.plot(y,'ro')
    #plt.xlabel('x')
    #plt.xlabel('y')
    #plt.show()
    cf_w = lR(x, y)
    plotLM(cf_w, x, y)
def bFLR(x, y, rank = 2):
    x = np.matrix(x)
    if x.shape[0] == 1:
        x = x.transpose()
    y = np.matrix(y)
    if y.shape[0] == 1:
        y = y.transpose()
    one = np.ones((x.shape[0], 1))
    tmp = np.zeros((x.shape[0], rank))
    for i in xrange(rank):
        tmp[:,i] = np.power(x.A, i + 1).transpose()
    xx = np.hstack([one, tmp])
    w = inv((xx.transpose()).dot(xx)).dot(np.transpose(xx)).dot(y)
    return w
def xlist(i, rank):
    l = [np.power(i ,ii) for ii in xrange(rank+1)]
    l = np.array([l]).transpose()
    return l
def plotBFLR(w, x, y, rank = 2):
    xx = [i for i in np.arange(0.0,1.0,1.0/30)]
    w = w.A.transpose()    
    yy = [w.dot(xlist(i, rank))[0,0] for i in xx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'ro')
    ax.plot(xx,yy)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str(rank) + ' order regression')
    plt.legend(('training sampes','regression line'))
    plt.show()
#regularization term on linear regression
def rTLR(x, y, lamda = 0.5,rank = 2):
    x = np.matrix(x)
    if x.shape[0] == 1:
        x = x.transpose()
    y = np.matrix(y)
    if y.shape[0] == 1:
        y = y.transpose()
    one = np.ones((x.shape[0], 1))
    tmp = np.zeros((x.shape[0], rank))
    for i in xrange(rank):
        tmp[:,i] = np.power(x.A, i + 1).transpose()
    xx = np.hstack([one, tmp])
    dim = xx.shape[1]
    I = lamda * np.diag(np.ones(dim))
    w = inv(I + (xx.transpose()).dot(xx)).dot(np.transpose(xx)).dot(y)
    return w
def plotRTLR(w, x, y, lamda, rank = 2):
    xx = [i for i in np.arange(0.0,1.0,1.0/20)]
    w = w.A.transpose()    
    yy = [w.dot(xlist(i, rank))[0,0] for i in xx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'ro')
    ax.plot(xx,yy)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('lambda = ' + str(lamda))
    plt.legend(('training sampes','regression line'))
    plt.show()
#multiple regression
def mxlist(i, rank):
    l = [np.power(i ,ii+1) for ii in xrange(rank+1)]
    l = np.array([l]).transpose()
    return l
def mrX(x, y, rank):
    a = mxlist(x, rank)
    b = mxlist(y, rank)
    one = np.ones((a.shape[0], 1))
    c = np.hstack([one, a[:,:,0], b[:,:,0]])
    return c
def mrLR(x, y, z, rank = 5):
    X = mrX(x, y, rank)
    z = np.matrix(z)
    if z.shape[0] == 1:
        z = z.transpose()
    w = inv((X.transpose()).dot(X)).dot(np.transpose(X)).dot(z)
    return w
def xyz(n = 20):
    t = np.arange(-3.0, 3.0, 6.0/n)
    [x, y] = np.meshgrid(t,t)
    z = x * x + y * y
    x = np.reshape(x,(x.shape[0] * x.shape[1], 1))
    y = np.reshape(y,(y.shape[0] * y.shape[1], 1))
    z = np.reshape(z,(z.shape[0] * z.shape[1], 1))
    
    return [x.ravel(), y.ravel(), z.ravel()]
def plotMR(w, x, y,z, rank = 5):
    X = mrX(x, y, rank)
    z = np.matrix(z)
    if z.shape[0] == 1:
        z = z.transpose()
    z = z.A
    z_p = X.dot(w)
    z_p = z_p.A
    
    xx = np.reshape(x, (20, 20))
    yy = np.reshape(y, (20, 20))
    zz = np.reshape(z, (20, 20))
    zz_p = np.reshape(z_p, (20, 20))
    plt.plot(zz.ravel())
    plt.plot(zz_p.ravel(), '.')
    plt.legend(('real data','prediction data'))
    plt.ylabel('z value')
    plt.xlabel('data point')
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.jet,
        #linewidth=0, antialiased=False)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #ax.plot_surface(xx, yy, zz_p, rstride=1, cstride=1,color='g',
        #linewidth=0, antialiased=False)
    plt.show()
#application
def xy(file_name):
    x = []
    y = []
    with open(file_name, 'r') as reader:
        for line in reader:
            ele = line.split(' ')
            x.append(float(ele[0]))
            y.append(float(ele[1]))
    plt.plot(x, y,'.')
    plt.show()
    return x, y
if __name__ == '__main__':
    #[x, y] = linearSamples()
    ##closed-form for linear regression
    #cf_w = lR(x, y)
    #plotLM(cf_w, x, y)
    ##gradient descient for linear regression
    #gd_w = gdLR(obj, x, y)
    #plotGdLM(cf_w, gd_w, x, y)
    ##basis function for linear regression
    [xnl, ynl] = nlSamples(2000)
    rank = 12
    bf_w = bFLR(xnl, ynl, rank)
    plotBFLR(bf_w, xnl, ynl, rank)
    #regularization
    rt_w = rTLR(xnl, ynl, lamda = 0.0000, rank = rank)
    writer = open('lamda.txt', 'w')
    for rt in rt_w:
        writer.write(str(rt[0,0])[0:5] + ' ')
    writer.close()
    plotRTLR(rt_w, xnl, ynl, lamda = 0.0000, rank = rank)
    #multiple regression
    r = 2
    [x, y, z] = xyz(n = 40)
    mr_w = mrLR(x, y, z, r)
    [tx, ty, tz] = xyz(n = 20)
    plotMR(mr_w, tx, ty, tz, r)
    
    #application
    [x, y] = xy('application.txt')
    
    
    
    
    
    