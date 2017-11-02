'''
Using SMO to train SVM.  The reference  paper: <<Sequential Minimal Optimization:
A Fast Algorithm for Training Support Vector Machines>>

Author: ZhongChao    superclocks@163.com

'''
import random
import numpy as np
from math import exp
import matplotlib
import matplotlib.pyplot as plt

class svm():
    def __init__(self,kernel = 'rbf',tolerance = 0.001,eps = 1.0e-12, C = 1.0, plot = True):
	self.kernel = kernel
	self.eps = eps
	self.C = C
	self.tolerance=tolerance
	self.alph_index = list()
	self.w = []
	self.plot = plot
	self.train_list = set()
	self.test_list = set()
    #d=2  #dimensins
    #self.train_num =10#the number of training samples
    
    #two_sigma_squared=2
   # b = 0.0
    #eps = 1.0e-12
    #C=1.0
    #alph = np.zeros((1,self.train_num))
    #training_matrix = np.zeros((self.train_num,2))
    #target_vector = np.zeros((1,self.train_num)) 
    #error_cache = np.zeros((1,self.train_num))
    #precomputed_self_dot_product = np.zeros((1,self.train_num))
    def cross_valindte(self, rate = 0.8):
	if(rate >= 1):
	    print 'The test sample should be less than the total samples!'
	    return 
	samples = len(self.training_matrix)
	while(True):
	    k0 = random.randint(0, samples - 1)
	    self.train_list.add(k0)
	    if len(self.train_list) / samples > rate:
		break
	    
    def smo(self, training_matrix,target_vector):
	self.training_matrix = training_matrix
	self.target_vector = target_vector
	self.d = training_matrix.shape[1]
	self.train_num = training_matrix.shape[0]
	self.two_sigma_squared = 2.0
	self.b = 0
	self.alph = np.zeros((1,self.train_num))
	self.error_cache = np.zeros((1,self.train_num))
	self.precomputed_self_dot_product = np.zeros((1,self.train_num))	

	#target = desired output vector
	#point = training point matrix
	num_changed = 0
	examine_all = 1
	self.initialize()
	while num_changed > 0 or examine_all == 1:
	    num_changed = 0
	    if(examine_all == 1):
		for k in range(0,self.train_num):
		    num_changed += self.examineExample(k)
	    else:
		for k in range(0,self.train_num):
		    if(self.alph[0][k] != 0 and self.alph[0][k] != self.C):
			num_changed += self.examineExample(k)
	    if(examine_all == 1):
		examine_all = 0
	    elif(not num_changed):
		examine_all = 1	
	for i in range(0, len(self.alph[0])):
	    if(self.alph[0][i] > self.eps):
		self.alph_index.append(i)
	
	print('-----------------------')
	print('sample number N=%d' %training_matrix.shape[0])
	print('self.train_num=%d' %self.train_num)
	#print('test_num=%d' %(N-self.train_num))
	print('demension d=%d' %self.training_matrix.shape[1])
	print('Threshold b=%f' %self.b)
	print('RBF kernel function\'s parameter two_sigma_squared = %r' % self.two_sigma_squared)	
    
    def calculateSum(self, j):
	s = 0
	for index in self.alph_index:
	    s = s + self.alph[0][index] * self.target_vector[0][index] * self.kernelFunc(index, j)
	return s
	
    #¼ÆËãb
    def calculateBias(self):
	b_new = 0
	for index in self.alph_index:
	    every_b = self.target_vector[0][index] - self.calculateSum(index) #¶ÔÓÚÃ¿¸ö·ÇÁãalphÇób
	    b_new = b_new + every_b
	self.b = b_new / len(self.alph_index)
    #def calculateOmiga(self):
	#w_list = 0
	#for index in self.alph_index:
	    #w_list = w_list + self.alph[0][index] * self.target_vector[0][index] * self.training_matrix[index, :]
	#for w in w_list:
	    #self.w.append(w)
    def plotSupportVector(self):
	#»­³öÈ«²¿ÑµÁ·Êý¾Ý
	matplotlib.rcParams['axes.unicode_minus'] = False
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(0,self.target_vector.shape[1]):
	    if(self.target_vector[0,i] == 1):
		ax.plot(self.training_matrix[i,0], self.training_matrix[i,1], 'r.',ms=10)
	    else:
		ax.plot(self.training_matrix[i,0], self.training_matrix[i,1], 'b.',ms=10)
	#»­³öÖ§³ÖÏòÁ¿
	for index in self.alph_index:
		ax.plot(self.training_matrix[index,0], self.training_matrix[index,1], 'o',ms=10, lw=2, alpha=0.7, mfc='white')
	ax.set_title('Suporrt Vector')
	plt.show()   	
	

    def train(self,training_matrix,target_vector):
	#µÚÒ»²½SMOËã·¨
	self.smo(training_matrix,target_vector)
	#µÚ¶þ²½ÓÃSMOÖÐÇóµÃµÄ·£²ÎÊý¼ÆËãb
	self.calculateBias()
	#»­³öÖ§³ÖÏòÁ¿
	if self.plot == True:
	    self.plotSupportVector()
    
    #def classify(self, data):
	 #self.training_matrix = data
	 
    def dotProductFunc(self,i1, i2):
	dot = 0.0
	for i in range(0,self.d):
	    dot = dot + self.training_matrix[i1][i] * self.training_matrix[i2][i]
	return dot
    
    def kernelFunc(self,i1, i2):
	s = self.dotProductFunc(i1, i2)
	s = s * -2
	s = s + self.precomputed_self_dot_product[0][i1] +  self.precomputed_self_dot_product[0][i2]
	return exp(-s / (self.two_sigma_squared * self.two_sigma_squared * 2))
    
    def learnedFunc(self,k):
	s = 0
	for i in range(0, self.train_num):
	    if(self.alph[0][i] > 0):
		s = s + self.alph[0][i] * self.target_vector[0][i] * self.kernelFunc(i, k)
	s = s - self.b
	return s
    

    def decisionFunc(self, vector):
	s = 0

    
	
    #ÓÃÓÚÓÅ»¯Á½¸ö³Ë×Ó£¬³É¹¦·µ»Ø1£¬·ñÔò·µ»Ø0
    def takeStep(self,i1, i2):
	if i1 == i2:
	    return 0
	
	alph1 = self.alph[0][i1]
	alph2 = self.alph[0][i2]
	y1 = self.target_vector[0][i1]
	y2 = self.target_vector[0][i2] 
	    
	if(alph1 > 0 and alph1 < self.C):
	    E1 = self.error_cache[0][i1]
	else:
	    E1 = self.learnedFunc(i1) - y1
	    
	if(alph2 > 0 and alph2 < self.C):
	    E2 = self.error_cache[0][i2]
	else:
	    E2 = self.learnedFunc(i2) - y2
	
	s = y1 * y2
	
	#¼ÆËã³Ë×ÓµÄÉÏÏÂÏÞ
	if (s == 1):
	    gamma = alph1 + alph2
	    if(gamma > self.C):
		L = gamma - self.C
		H = self.C
	    else:
		L = 0
		H = gamma
	else:
	    gamma = alph1 - alph2
	    if(gamma > 0):
		L = 0
		H = self.C - gamma
	    else:
		L = -gamma
		H = self.C
	if(L == H):
	    return 0
	 
	#¼ÆËãeta
	k11 = self.kernelFunc(i1, i1)
	k22 = self.kernelFunc(i2, i2)
	k12 = self.kernelFunc(i1, i2)
	eta = 2 * k12 - k11 - k22
	if(eta < -0.001):
	    c = y2 * (E2 - E1)
	    a2 = alph2 + c / eta #¼ÆËãÐÂµÄalph2
	    if(a2 < L):
		a2 = L
	    elif(a2 > H):
		a2 = H            
	else: #·Ö±ð´Ó¶ÏµãH,LÇóÄ¿±êº¯ÊýLobj,Hobj£¬È»ºóÉèa2ÎªËùÇóµÃ×î´óÄ¿±êº¯ÊýÖµ
	    c1 = eta / 2
	    c2 = y2 *(E1 - E2) - eta * alph2
	    Lobj = c1 * L * L + c2 * L
	    Hobj = c1 * H * H + c2 * H
	    if(Lobj > Hobj + self.eps):
		a2 = L
	    elif(Hobj > Lobj + self.eps):
		a2 = H
	    else:
		a2 = alph2
	
	if(abs(a2 - alph2) < self.eps):
	    return 0  
	
	a1 = alph1 - s * (a2 - alph2) #¼ÆËãÐÂµÄa1
	
	if(a1 < 0):
	    a2 = a2 + s * a1
	    a1 = 0
	elif(a1 > self.C):
	    a2 = a2 + s * (a1 - self.C)
	    a1 = self.C
	
	#¸üÐÂãÐÖµb
	#global b
	if(a1 > 0 and a1 < self.C):
	    bnew = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
	else:
	    if(a2 > 0 and a2 < self.C):
		bnew = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
	    else:	    
		b1 = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
		b2 = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
		bnew = (b1 + b2) / 2
		
	delta_b = bnew - self.b;
	self.b = bnew;
	
	#¶ÔÓÚÏßÐÔÇé¿ö£¬Òª¸üÐÂÈ¨ÏòÁ¿£¬ÕâÀï²»ÓÃÁË 
	#¸üÐÂerror_cache£¬¶ÔÓÚ¸üÐÂºóµÄa1,a2,¶ÔÓ¦µÄÎ»ÖÃi1,i2µÄerror_cache[i1] = error_cache[i2] = 0 
	t1 = y1 * (a1 - alph1);
	t2 = y2 * (a2 - alph2);
	for i in range(0,self.train_num):
	    if(self.alph[0][i] > 0 and self.alph[0][i] < self.C):
		    self.error_cache[0][i] += t1 * self.kernelFunc(i1,i) + t2 * (self.kernelFunc(i2,i)) - delta_b
	self.error_cache[0][i1] = 0
	self.error_cache[0][i2] = 0
	self.alph[0][i1] = a1
	self.alph[0][i2] = a2 #´æ´¢a1,a2µ½Êý×é 
	return 1
    
    #ÔÚnon-bound³Ë×ÓÖÐÑ°ÕÒmaximum abs(E1-E2)µÄÑù±¾
    def examineFirstChoice(self,i1, E1):
	tmax = 0
	i2 = -1
	for k in range(0, self.train_num-1):
	    if(self.alph[0][k] <0.00000001 or self.alph[0][k] == self.C):
		E2 = self.error_cache[0][k]
		temp = abs(E1 - E2)
		if(temp > tmax):
		    tmax = temp
		    i2 = k
	if(i2 >= 0 and self.takeStep(i1, i2)):
	    return 1
	return 0
    
    #Èç¹ûÉÏÃæÃ»ÓÐ½øÕ¹£¬ÄÇÃ´´ÓËæ»úÎ»ÖÃ²éÕÒnon-bountÑù±¾
    def examineNonBound(self,i1):
	#k0=random.random()%self.train_num
	while 1:
	    #k0=random.randint(0,100000000)%self.train_num
	    k0 = random.randint(0,len(self.alph[0])-1)
	    if k0 != i1:
		break
	    
	#print("2:%d" %k0)
	for k in range(0,self.train_num):
	    i2 = (k+k0)%self.train_num
	    if((self.alph[0][i2] == 0 or self.alph[0][i2] == self.C) and self.takeStep(i1,i2)):
		return 1
	return 0
    
    #Èç¹ûÉÏÃæÒ²Ê§°Ü£¬Ôò´ÓËæ»úÎ»ÖÃ²éÕÒÕû¸öÑù±¾£¨¸ÄÎªboundÑù±¾£©
    def examineBound(self,i1):
	#k0 = rand() % self.train_num
	while 1:
	    #k0=random.randint(0,100000000)%self.train_num
	    k0 = random.randint(0,len(self.alph[0])-1)
	    if k0 != i1:
		break
	    
	#print("3:%d" %k0)
	for k in range(0, self.train_num):
	    i2 = (k + k0) % self.train_num;
	    if(self.takeStep(i1,i2)):
		return 1;
	return 0;
	
    def examineExample(self,i1):
	y1 = self.target_vector[0][i1]
	    
	alph1 = self.alph[0][i1]
	if(alph1 > 0 and alph1 < self.C):
	    E1 = self.error_cache[0][i1]
	else:
	    E1 = self.learnedFunc(i1) - y1
	r1 = y1 * E1
	if((r1 > self.tolerance and alph1 > 0) or (r1 < -self.tolerance and alph1 < self.C)):
	    #Ê¹ÓÃÈýÖÖ·½·¨Ñ¡ÔñµÚ¶þ¸ö³Ë×Ó 
	    #1£ºÔÚnon-bound³Ë×ÓÖÐÑ°ÕÒmaximum fabs(E1-E2)µÄÑù±¾ 
	    #2£ºÈç¹ûÉÏÃæÃ»È¡µÃ½øÕ¹,ÄÇÃ´´ÓËæ»úÎ»ÖÃ²éÕÒnon-boundary Ñù±¾ 
	    #3£ºÈç¹ûÉÏÃæÒ²Ê§°Ü£¬Ôò´ÓËæ»úÎ»ÖÃ²éÕÒÕû¸öÑù±¾,¸ÄÎªboundÑù±¾ 
	    if(self.examineFirstChoice(i1, E1)):#µÚ1ÖÖÇé¿ö 
		return 1
	    if(self.examineNonBound(i1)): #µÚ2ÖÖÇé¿ö 
		return 1
	    if(self.examineBound(i1)): #µÚ3ÖÖÇé¿ö
		return 1     
	return 0
    
    #¼ÆËãÎó²îÂÊ
    def errorRate(self):
	ac = 0
	print("-------------²âÊÔ½á¹û------------")
	for i in range(0, self.train_num):
	    tar = learnedFunc(i)
	    if((tar > 0 and targe[i]) > 0 or (tar < 0 and target[i] < 0)):
		ac = ac + 1
	accuracy = ac / (N - self.train_num)
	print("¾«È·¶È£º",accuracy * 100)
    
    def dataSource(self):
	import matplotlib.pyplot as plt
	import string
	
	m = 5
	n = 5
	#training_matrix[0:m,0] = [random.uniform(0,1) for i in range(0,m)]
	#training_matrix[0:m,1] = [random.uniform(0,1) for i in range(0,m)]
	#target_vector[0][0:m] = [-1 for i in range(0,m)]
	
	#training_matrix[m:m+n,0] = [random.uniform(3,4) for i in range(0,n)]
	#training_matrix[m:m+n,1] = [random.uniform(3,4) for i in range(0,n)]
	#target_vector[0][m:m+n] = [1 for i in range(0,n)]    
	
	###pl.figure(1)
	###pl.clf()
	#write = file('training_matrix.txt','w')
	#for i in range(0, m+n):	
	    #write.write(str(target_vector[0][i]))
	    #write.write(' ')
	    #write.write(str(training_matrix[i][0]))
	    #write.write(' ')
	    #write.write(str(training_matrix[i][1]) + '\n')
	
	#write.close()
	
	read = file('training_matrix.txt','r')
	for i in range(0,m+n):
	    x = read.readline().split(' ')
	    target_vector[0][i] = string.atof(x[0])
	    for j in range(1,3):
		training_matrix[i][j-1] = string.atof(x[j])
	read.close()
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.plot(training_matrix[0:100,0],training_matrix[0:100,1],'.')
	#ax.plot(training_matrix[100:200,0],training_matrix[100:200,1],'r.')
	#plt.show()    
    def initialize(self):
	
	#dataSource()
	self.precomputed_self_dot_product[0,:] = [self.dotProductFunc(i,i) 
	                                      for i in range(0, self.train_num)]
    


if __name__ == "__main__":    
    
    import string
	    
    m = 5
    n = 5
    training_matrix = np.zeros((150,2))
    target_vector = np.zeros((1,150))
    read = file('t.txt','r')
    for i in range(0,150):
	x = read.readline().split('\t')
	target_vector[0][i] = string.atof(x[0])
	for j in range(1,3):
	    training_matrix[i][j-1] = string.atof(x[j])
    read.close()    
    svm().train(training_matrix,target_vector)
    
    #train()
    
    #initialize()
    #dataSource()
    #training = np.zeros((2,3))
    #target = np.zeros((1,2))
    #errorcache = np.zeros((1,target.shape[1]))
    #tolerance = 0.001
    #d = training.shape[1]
    #precomputed_self_dot_product = np.zeros((1,training.shape[1]))
    #smo(training, target);
    
   
    
    