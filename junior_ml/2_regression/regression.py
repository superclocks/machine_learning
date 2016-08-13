import numpy as np
import matplotlib.pyplot as plt

from synthic_data import linearSamples


if __name__ == '__main__':
    data = linearSamples()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot(data, '.')
    plt.show()