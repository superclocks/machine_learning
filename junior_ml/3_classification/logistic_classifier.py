import matplotlib.pyplot as plt
import numpy as np
import pylab

def logiDist():
    mu = 0.5
    gama = 0.3
    x = np.arange(-5.0, 5.0, 10.0/100)
    dist = 1.0 / (1.0 + np.exp(-(x - mu)/gama))
    dens = np.exp(-(x - mu)/gama) / (gama * np.power((1.0 + np.exp(-(x - mu)/gama)), 2))
    
    fig = plt.figure(1)
    ax = fig.add_subplot(211)
    ax.plot(x, dist)
    ax.plot([mu, mu],[0.0, 1.0])
    plt.xlabel('X')
    plt.ylabel('F(X)')
    
    pylab.text(1.05, 0.5, r"$\mu=%s,\gamma=%s$"%(str(mu),str(gama)), {'color' : 'g', 'fontsize' : 20},
           horizontalalignment = 'left',
           verticalalignment = 'center',
           rotation = 0,
           clip_on = False)
    
    ax = fig.add_subplot(212)
    ax.plot(x, dens)
    ax.plot([mu, mu],[0.0, 1.0])
    plt.xlabel('X')
    plt.ylabel('f(X)')
    pylab.text(1.05, 0.5, r"$\mu=%s,\gamma=%s$"%(str(mu),str(gama)), {'color' : 'g', 'fontsize' : 20},
           horizontalalignment = 'left',
           verticalalignment = 'center',
           rotation = 0,
           clip_on = False)
    plt.show()
if __name__ == '__main__':
    logiDist()