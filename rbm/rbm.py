import matplotlib.pylab as plt
import numpy as np
import random
from scipy.linalg import norm
import PIL.Image


class Rbm:
    def __init__(self,n_visul, n_hidden, max_epoch = 50, batch_size = 110, penalty = 2e-4, anneal = False, w = None, v_bias = None, h_bias = None):
        self.n_visible = n_visul
        self.n_hidden = n_hidden
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty
        self.anneal = anneal
        
        if w is None:
            self.w = np.random.random((self.n_visible, self.n_hidden)) * 0.1
        if v_bias is None:
            self.v_bias = np.zeros((1, self.n_visible))
        if h_bias is None:
            self.h_bias = np.zeros((1, self.n_hidden))
    def sigmod(self, z):
        return 1.0 / (1.0 + np.exp( -z ))

    def forward(self, vis):
        #if(len(vis.shape) == 1):
            #vis = np.array([vis])
	    #vis = vis.transpose()
        #if(vis.shape[1] != self.w.shape[0]):
	vis = vis.transpose()
        
        pre_sigmod_input = np.dot(vis, self.w) + self.h_bias
        return self.sigmod(pre_sigmod_input)
    
    def backward(self, vis):
        #if(len(vis.shape) == 1):
            #vis = np.array([vis])
	    #vis = vis.transpose()
        #if(vis.shape[0] != self.w.shape[1]):
        back_sigmod_input = np.dot(vis, self.w.transpose()) + self.v_bias
        return self.sigmod(back_sigmod_input)
    def batch(self):
        
        eta = 0.1
        momentum = 0.5
	d, N = self.x.shape
        
        num_batchs = int(round(N / self.batch_size)) + 1
        groups = np.ravel(np.repeat([range(0, num_batchs)], self.batch_size, axis = 0))
        groups = groups[0 : N]
        perm = range(0, N)
        random.shuffle(perm)
        groups = groups[perm]
        batch_data = []
        for i in range(0, num_batchs):
            index = groups == i
            batch_data.append(self.x[:, index])
        return batch_data
    def rbmBB(self, x):
	self.x = x
	eta = 0.1
	momentum = 0.5
	W = self.w
	b = self.h_bias
	c = self.v_bias
	Wavg = W
	bavg = b
	cavg = c
	Winc  = np.zeros((self.n_visible, self.n_hidden))
	binc = np.zeros(self.n_hidden)
	cinc = np.zeros(self.n_visible)
	avgstart = self.max_epoch - 5;
        batch_data = self.batch()
        num_batch = len(batch_data)
        
        oldpenalty= self.penalty
	t = 1
	errors = []
        for epoch in range(0, self.max_epoch):
            err_sum = 0.0
            if(self.anneal):
                penalty = oldpenalty - 0.9 * epoch / self.max_epoch * oldpenalty
            
            for batch in range(0, num_batch):
                num_dims, num_cases = batch_data[batch].shape
                data = batch_data[batch]
                #forward
                ph = self.forward(data)
                ph_states = np.zeros((num_cases, self.n_hidden))
                ph_states[ph > np.random.random((num_cases, self.n_hidden))] = 1
                
                #backward
                nh_states = ph_states
                neg_data = self.backward(nh_states)
                neg_data_states = np.zeros((num_cases, num_dims))
                neg_data_states[neg_data > np.random.random((num_cases, num_dims))] = 1
                
                #forward one more time
		neg_data_states = neg_data_states.transpose()
                nh = self.forward(neg_data_states)
                nh_states = np.zeros((num_cases, self.n_hidden))
                nh_states[nh > np.random.random((num_cases, self.n_hidden))] = 1
		
                #update weight and biases
                dW = np.dot(data, ph) - np.dot(neg_data_states, nh)
                dc = np.sum(data, axis = 1) - np.sum(neg_data_states, axis = 1)
                db = np.sum(ph, axis = 0) - np.sum(nh, axis = 0)
                Winc = momentum * Winc + eta * (dW / num_cases - self.penalty * W)
                binc = momentum * binc + eta * (db / num_cases);
		cinc = momentum * cinc + eta * (dc / num_cases);
		W = W + Winc
		b = b + binc
		c = c + cinc
		
		self.w = W
		self.h_bais = b
		self.v_bias = c
		if(epoch > avgstart):
		    Wavg -= (1.0 / t) * (Wavg - W)
		    cavg -= (1.0 / t) * (cavg - c)
		    bavg -= (1.0 / t) * (bavg - b)
		    t += 1
		else:
		    Wavg = W
		    bavg = b
		    cavg = c
		#accumulate reconstruction error
		err = norm(data - neg_data.transpose())

		err_sum += err
	    print epoch, err_sum
	    errors.append(err_sum)
	self.errors = errors
	self.hiden_value = self.forward(self.x)
	
	h_row, h_col = self.hiden_value.shape
	hiden_states = np.zeros((h_row, h_col))
	hiden_states[self.hiden_value > np.random.random((h_row, h_col))] = 1
	self.rebuild_value = self.backward(hiden_states)
	
	self.w = Wavg
	self.h_bais = b
	self.v_bias = c
    def visualize(self, X):
	D, N = X.shape
	s = int(np.sqrt(D))
	if s == int(np.floor(s)):
	    num = int(np.ceil(np.sqrt(N)))
	    a = np.zeros((num*s + num + 1, num * s + num + 1)) - 1.0
	    x = 0
	    y = 0
	    for i in range(0, N):
		z = X[:,i]
		z = z.reshape(s,s,order='F')
		
		z = z.transpose()
		a[x*s+1+x - 1:x*s+s+x , y*s+1+y - 1:y*s+s+y ] = z
		x = x + 1
		if(x >= num):
		    x = 0
		    y = y + 1
	    d = True
	else:
	    a = X
	return a
def readData(path):
    data = []
    for line in open(path, 'r'):
	ele = line.split(' ')
	tmp = []
	for e in ele:
	    if e != '':
		tmp.append(float(e.strip(' ')))
	data.append(tmp)
    return data

if __name__ == '__main__':
    data = readData('data.txt')
    data = np.array(data)
    data = data.transpose()
    rbm = Rbm(784, 100,max_epoch = 50)
    rbm.rbmBB(data)
    
    a = rbm.visualize(data)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(a)
    plt.title('original data')
    
    rebuild_value = rbm.rebuild_value.transpose()
    b = rbm.visualize(rebuild_value)
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.imshow(b)
    plt.title('rebuild data')
    
    hidden_value = rbm.hiden_value.transpose()
    c = rbm.visualize(hidden_value)
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.imshow(c)
    plt.title('hidden data')

    w_value = rbm.w
    d = rbm.visualize(w_value)
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.imshow(d)
    plt.title('weight value(w)')
    plt.show()

    
    
    
    