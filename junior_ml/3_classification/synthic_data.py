import numpy as np
import random
import math
import matplotlib.pyplot as plt
def twoSamples(n = 30):
    x = []
    y = []
    x.extend([[1.0 + random.random(), 1.0 + random.random()] for i in xrange(n)])
    x.extend([[2.5 + random.random(), 2.5 + random.random()] for i in xrange(n)])
    y.extend([-1 for i in xrange(n)])
    y.extend([1 for i in xrange(n)])
    
    #plt.plot(a[:,0], a[:,1],'o')
    #plt.plot(b[:,0], b[:,1],'ro')
    #plt.xlim([0, 5])
    #plt.ylim([0, 5])
    #plt.show()
    return [x, y]
    

if __name__ == '__main__':
    [x, y] = twoSamples()