from scipy.spatial.distance import pdist, cdist
from math import exp,sqrt
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from kmeans import kmeans
import copy
import matplotlib.pylab as plt


def readData(path):
    data = []
    for line in open(path, 'r'):
        ele = line.split('\t')
        tmp = []
        for e in ele:
            tmp.append(float(e))
        data.append(tmp)
    return data

def normalize(data):
    mean = np.mean(data, axis = 0)
    data = (data - mean)
    max_ = np.max(np.abs(data))
    data = data / max_
    return data
    
def affinity(data, knn):
    data = np.array(data)
    
    d = cdist(data,data,'sqeuclidean')
    
    sigma_list = np.zeros(data.shape[0])
    dis_matrix = np.zeros((data.shape[0], data.shape[0]))
    i = 0
    for e in d:
        ec = copy.deepcopy(e)
        ec.sort()
        index = ec > 0.0
        tmp = ec[index]
        sigma_list[i] = tmp[knn]
        i = i + 1
    #save(dis_matrix, 'dis_matrix.txt')
    dis_matrix = updataAffinity(d, sigma_list)
    return dis_matrix           
    
def updataAffinity(dis_matrix, sigma_list):
    row = dis_matrix.shape[0]
    col = row
    for i in range(0, row):
        for j in range(0, col):
            dis_matrix[i, j] = exp(-dis_matrix[i, j]/(sigma_list[i] * sigma_list[j]))#(
            
    for i in range(0, row):
        dis_matrix[i, i] = 0.0
    return dis_matrix
                
def dMatrix(dis_matrix):
    _sum = sum(dis_matrix)
    _sqrt = np.sqrt(_sum)
    d = 1.0 / _sqrt
    return d
def calVector(str):
    r = []
    str = str[1:len(str) - 2]
    ele = str.split(',')
    for e in ele[0: len(ele)]:
        r.append(float(e))
    return r

def lMatrix(dis_matrix):
    d = dMatrix(dis_matrix)
    if (len(d.shape) == 1):
        d = np.array([d])
    if(d.shape[0] == 1):
        d_row = copy.deepcopy(d.transpose())
        d_col = copy.deepcopy(d)
    else:
        d_row = copy.deepcopy(d)
        d_col = copy.deepcopy(d.transpose())
    
    tmp = d_col * dis_matrix
    l = tmp * d_row
    return l

def svd(l):
    u,s,v = svds(l,2)
    return u
            
def unit(data):
    norm2 = np.array([[norm(di) for di in data]]).transpose()
    u = data / norm2
    return u
#def kmeans(data, n_clusters):
    #km = KMeans(n_clusters = n_clusters)
    #km.fit(data)
    #label = km.labels_
    #return label



if __name__ == '__main__':   

    #data = readData_1('dtz_2week.dat',8)
    data = readData('kmeans_spectral.txt')
    #plt.plot(np.array(data)[:, 0], np.array(data)[:, 1],'.r')
    #plt.show()
    data = normalize(data)
    k_sigma = 100
    dis_matrix = affinity(data, k_sigma)
    
    l = lMatrix(dis_matrix)
    print 'Calculate SVD.'
    u = svd(l)
    print 'Normalize the u.'
    unit_dat = unit(u)
    plt.plot(unit_dat[:, 0], unit_dat[:, 1],'r.')
    plt.show()
    print 'Apply kmeans on U matrix.'
    cluster_label = kmeans(unit_dat, k = 3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ['r','g','b']
    for e in set(cluster_label):
        index = cluster_label == e
        dd = data[index, :]
        ax.plot(dd[:, 0], dd[:, 1],'.'+c[e])
    plt.legend(('label1','label2','label3'))
    plt.show()
        
        
        
        
        