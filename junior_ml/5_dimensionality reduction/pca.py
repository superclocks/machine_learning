import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eig

def samples(file_name):
    d = []
    for line in open(file_name):
        ele = line.split('\t')
        tmp = []
        for e in ele:
            tmp.append(int(e))
        d.append(tmp)
    d = np.array(d)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot(d[:, 0], d[:, 1], d[:,2], 'o')
    #ax.set_xlabel('story')
    #ax.set_ylabel('music')
    #ax.set_zlabel('performer')
    #plt.show()
    return d
def unit(data):
    mean_ = np.mean(data, axis = 0)
    std_ = np.std(data, axis = 0, ddof = 1)
    return (data - mean_) / std_
def cov(data):  
    mean_ = np.mean(data, axis = 0)  
    data = data - mean_  
    cov_mat = data.T.dot(data) / (data.shape[0] - 1)  
    return cov_mat  
def plotItems(u):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.abs(u[:, 0]), u[:, 1], 'o')
    u[:, 0] = np.abs(u[:, 0])
    plt.text(u[0, 0], u[0, 1], 'story')
    plt.text(u[1, 0], u[1, 1], 'music')
    plt.text(u[2, 0], u[2, 1], 'performer')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.show()
def samplesPca(data, u):
    z1 = np.sum(data * np.abs(u[:, 0]), axis = 1)
    z2 = np.sum(data * u[:, 1], axis = 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    markerline, stemlines, baseline = ax.stem(z1, z2, '-.')
    for i in range(0,z1.shape[0]):
        plt.text(z1[i], z2[i], str(i+1))
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    u[:, 0] = np.abs(u[:, 0])
    ax.plot(u[:, 0], u[:, 1], '*')
    plt.text(u[0, 0], u[0, 1], 'story')
    plt.text(u[1, 0], u[1, 1], 'music')
    plt.text(u[2, 0], u[2, 1], 'performer')
    plt.show()
if __name__ == '__main__':
    d = samples('pca_demo.txt')
    #1)unit data
    u_d = unit(d)
    
    #2)covariance matrix
    cov_matrix = cov(u_d)
    #3)
    [u, s] = eig(cov_matrix)
    #4)
    plotItems(s)
    #5)
    samplesPca(u_d, s)
    
    
    
    
    
    
    
    
    