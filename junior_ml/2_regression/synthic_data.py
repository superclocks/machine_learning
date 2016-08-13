import numpy as np
import random
import math
import matplotlib.pyplot as plt
def linearSamples(n = 20):
    a = 0.5 
    b = 1.0
    r = [i + 2.0*random.random() for i in xrange(n)]
    return [range(0, len(r)), r]

def nlSamples(n = 100):
    t = np.arange(0, 1.0, 1.0 / n)
    y = [ti + 0.3 * math.sin(2 * math.pi * ti)+random.random()*0.0  for ti in t]
    t = list(t)
    return [t, y]
if __name__ == '__main__':
    nlSamples()
    