from scipy.spatial.distance import pdist
from math import exp,sqrt
import numpy as np
from scipy.linalg import norm
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs


def readData(path):
    data = []
    for line in open(path, 'r'):
        ele = line.split('\t')
        tmp = []
        for e in ele:
            tmp.append(float(e))
        data.append(tmp)
    return data
def observer(iter, X, labels, centers):
        print "iter %d." % iter
        centers = np.array(centers)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pyplot.plot(hold=False)  # clear previous plot
        pyplot.hold(True)
 
        # draw points
        data_colors=[colors[lbl] for lbl in labels]
        pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        # draw centers
        pyplot.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)
        pyplot.title("iter %d." % iter)
        pyplot.show(block=False)
        pyplot.savefig('kmeans_fig/iter_%02d.png' % iter, format='png')
        
def kmeans(data, k, tol = 1e-6):
    data = np.array(data)
    import random 
    index = np.arange(0, len(data))
    random.shuffle(index)
    index = index[0: k]
    #(1)
    init_center = data[index, :]
    labels = np.zeros((data.shape[0],),dtype=np.int)
    iter_ = 0
    while iter_ < 100 :
        err = 0.0
        #allocate the sample to the one center whitch is closed that sample
        for i in xrange(data.shape[0]):
            di = data[i, :]
            max_dist = 1e100
            tmp = []
            for center in init_center:
                dis = pdist([di, center])
                tmp.append(dis[0])
            labels[i] = np.argmin(tmp)
        #calculate the new centers
        new_center = []
        for label in set(labels):
            label_data = data[labels == label, :]
            new_center.append(np.sum(label_data, axis=0) / len(label_data))
        #calculate the difference between old center and new center
        for i in range(0, k):
            err = err + norm(new_center[i] - init_center[i])
        init_center = new_center
        iter_ = iter_ + 1
        observer(iter_, data, labels, new_center)
        if err < tol:
            break
    return labels

if __name__ == '__main__':   
    data = readData('kmeans_data.txt')
    data = np.array(data)
    data = data[:, 0:2]
    cluster_label = kmeans(data, k = 3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ['r','g','b']
    for e in set(cluster_label):
        index = cluster_label == e
        dd = data[index, :]
        ax.plot(dd[:, 0], dd[:, 1],'.'+c[e])
    plt.show()
        
        
        
        
        