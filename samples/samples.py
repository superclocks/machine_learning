import math
import time
import numpy as np
import matplotlib.pylab as plt
import random
class Samples:
    def __init__(self):
        pass
    def rand(self, num, seed = 1):
        m = math.pow(2, 32)
        a = 214013
        c = 2531011
        i = 1
        x = np.zeros(num)
        x[0] = seed
        while(i < num):
            x[i] = (a * x[i-1] + c) % m
            i += 1
        return x
    def uniform(self, num, seed = 1):
        m = math.pow(2, 32)
        x = self.rand(num, seed)
        return x / m
    def normal(self, num):
        t = int(time.time()) 
        u1 = self.uniform(num, t)
        t1 = int(time.time())+1
        u2 = self.uniform(num, t1)
        z0 = np.sqrt(-2 * np.log(u2)) * np.sin(2.0 * np.pi * u1)
        #plt.hist(z0)
        #plt.show()
        return z0
    def acceptanceRejection(self, distri_style): #Acceptance-Rejection sampling
        if distri_style == 'normal':
            a = -10
            b = 10
            i = 0
            z = []
            while i < 1000000:
                u1 = math.exp(- (a + (b - a)*np.random.uniform(0,1,1)))
                u2 = math.exp(- (a + (b - a)*np.random.uniform(0,1,1)))
                y1 = - math.log(u1)
                y2 = - math.log(u2)
                if(y2 > math.pow(y1-1,2)/2):
                    z.append(y1)
                    i += 1
            plt.hist(z,np.arange(a,b,0.5))
            plt.show()
    def mh(self, epsilon_0, num_iteration, fpdf):
        #Metropolis¨CHastings algorithm
        normal_randoms = np.zeros(num_iteration)
        uniform_randoms = np.zeros(num_iteration)
        for i in range(0, num_iteration):
            uniform_randoms[i] = random.uniform(0, 1)
            normal_randoms[i] = random.normalvariate(0, 1)
        #fig = plt.figure()
        #ax = fig.add_subplot(211)
        #ax.plot(normal_randoms, '.')
        #ax1 = fig.add_subplot(212)
        #ax1.plot(uniform_randoms, '.')
        #plt.show()
        
        epsilon = np.zeros(num_iteration)
        previous_epsilon = epsilon_0
        for i in range(0, num_iteration):
            epsilon_tilde = previous_epsilon + normal_randoms[i]
            if(fpdf(epsilon_tilde) > fpdf(previous_epsilon)):
                epsilon[i] = epsilon_tilde
            else:
                if(uniform_randoms[i] <= fpdf(epsilon_tilde) / fpdf(previous_epsilon)):
                    epsilon[i] = epsilon_tilde
                else:
                    epsilon[i] = previous_epsilon
            previous_epsilon = epsilon[i]
        return epsilon
    
    def mh1(self, epsilon_0, num_iteration, fpdf):
        #Metropolis¨CHastings algorithm
        normal_randoms = np.zeros(num_iteration)
        uniform_randoms = np.zeros(num_iteration)
        for i in range(0, num_iteration):
            uniform_randoms[i] = random.uniform(0, 1)
            normal_randoms[i] = random.normalvariate(0, 1)
        
        epsilon = np.zeros(num_iteration)
        previous_epsilon = epsilon_0
        for i in range(0, num_iteration):
            epsilon_tilde = previous_epsilon + normal_randoms[i]
            rate = fpdf(epsilon_tilde) / fpdf(previous_epsilon)
            alfa = min(rate, 1.0)
            if(uniform_randoms[i] < alfa):
                epsilon[i] = epsilon_tilde
            else:
                epsilon[i] = previous_epsilon
            previous_epsilon = epsilon[i]
        return epsilon
def nor(x):
    return (1.0/np.sqrt(2.0*np.pi))*np.exp(-np.power(x,2)/2)
if __name__=='__main__':
    s = Samples()
    #x00 = s.rand(1000)
    #x00 = s.uniform(1000)
    #x00 = s.normal(1000)
    #plt.hist(x00,100)
    #plt.show()
    x0 = s.mh(0, 5000, nor)
    
    x = s.mh1(0, 5000, nor)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(x0,200)
    plt.title('mh')
    ax = fig.add_subplot(212)
    ax.hist(x, 200)
    plt.title('mh1')
    plt.show()
    s.acceptanceRejection('normal')