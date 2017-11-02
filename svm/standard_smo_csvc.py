import math
import pickle
import random
 
from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from ..nlp.segmenter import Segmenter
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration
 
class StandardSMO:
    '''Platt's standard SMO algorithm for csvc.'''
    
    def __init__(self,config, nodeName, loadFromFile = False, C = 100, tolerance = 0.001):
        #store a number nearly zero.
        self.accuracy = 1E-3
        #store penalty coefficient of slack variable.
        self.C = C
        #store tolerance of KKT conditions.
        self.tolerance = tolerance        
        #store isTrained by data.
        self.istrained = loadFromFile
 
        #store lagrange multipiers.
        self.alpha = []
        #store weight
        self.w = []
        #store threshold.
        self.b = float(0)        
        #store kii
        self.kcache = {}
        
        self.istrained = False
 
        #-------------------begin model info-------------------------------
        self.curNode = config.GetChild(nodeName)
        self.modelPath = self.curNode.GetChild("model_path").GetValue()
        self.logPath = self.curNode.GetChild("log_path").GetValue()
        #-------------------end  model info-------------------------------
 
        #-------------------begin kernel info-------------------------------
        self.curNode = config.GetChild(nodeName)
        self.kernelNode = self.curNode.GetChild("kernel");
        self.kernelName = self.kernelNode.GetChild("name").GetValue();
        #to get parameters from top to button -> from left to right -> from inner to outer.        
        self.parameters = self.kernelNode.GetChild("parameters").GetValue().split(',');
        #-------------------end  kernel info-------------------------------
 
        if (loadFromFile):
            f = open(self.modelPath, "r")
            modelStr = pickle.load(f)
            [self.alphay, self.sv, self.b, self.w] = pickle.loads(modelStr)
            f.close()
            self.istrained = True
 
    def DotProduct(self,i1,i2):
        '''To get vector's dot product for training.'''
 
        dot = float(0)
        for i in range(0,self.trainx.nCol):
            dot += self.trainx.Get(i1,i) * self.trainx.Get(i2,i)  
        return dot
        
    def Kernel(self):
        '''To get kernel function with configuration for training.
 
            kernel function includes RBF,Linear and so on.'''      
 
        if self.kernelName == 'RBF':
            return lambda xi,yi: math.exp((2*self.DotProduct(xi,yi)-self.DotProduct(xi,xi)-self.DotProduct(yi,yi))/(2*float(self.parameters[0])*float(self.parameters[0])))
        elif self.kernelName == 'Linear':
            return lambda xi,yi:self.DotProduct(xi,yi) + float(self.parameters[0])
        elif self.kernelName == 'Polynomial':
            return lambda xi,yi: (float(self.parameters[0]) * self.DotProduct(xi,yi) + float(self.parameters[1])) ** int(self.parameters[2])
    
    def DotVectorProduct(self,v1,v2):
        '''To get vector's dot product for testing.'''
 
        if len(v1) != len(v2):
            print 'The dimensions of two vector should equal'
            return 0.0
        dot = float(0)
        for i in range(0,len(v1)):
            dot += v1[i] * v2[i]
        return dot
        
    def KernelVector(self, v1, v2):
        '''To get kernel function for testing.'''
        
        if self.kernelName == 'RBF':
            return math.exp((2*self.DotVectorProduct(v1, v2)-self.DotVectorProduct(v1, v1)-self.DotVectorProduct(v2, v2))/(2*float(self.parameters[0])*float(self.parameters[0])))
        elif self.kernelName == 'Linear':
            return self.DotVectorProduct(v1, v2) + float(self.parameters[0])
        elif self.kernelName == 'Polynomial':
            return (float(self.parameters[0]) * self.DotVectorProduct(v1,v2) + float(self.parameters[1])) ** int(self.parameters[2])
        
    def F(self,i1):
        '''To calculate output of an sample.
 
            return output.'''
                
        if self.kernelName == 'Linear':
            dot = 0
            for i in range(0,self.trainx.nCol):
                dot += self.w[i] * self.trainx.Get(i1,i);    
            return dot + self.b
 
        K = self.Kernel()   
        final = 0.0
        for i in range(0,len(self.alpha)):
            if self.alpha[i] > 0:
                key1 = '%s%s%s'%(str(i1), '-', str(i))
                key2 = '%s%s%s'%(str(i), '-', str(i1))
                if self.kcache.has_key(key1):
                    k = self.kcache[key1]
                elif self.kcache.has_key(key2):
                    k = self.kcache[key2]
                else:
                    k =  K(i1,i)
                    self.kcache[key1] = k
                    
                final += self.alpha[i] * self.trainy[i] * k
        final += self.b
        return final
 
    def examineExample(self,i1):
        '''To find the first lagrange multipliers.
 
                then find the second lagrange multipliers.'''
        y1 = self.trainy[i1]
        alpha1 = self.alpha[i1]
 
        E1 = self.F(i1) - y1
 
        kkt = y1 * E1
 
        if (kkt > self.tolerance and kkt > 0) or (kkt <- self.tolerance and kkt < self.C):#not abide by KKT conditions
            if self.FindMaxNonbound(i1,E1):
                return 1
            elif self.FindRandomNonbound(i1):
                return 1
            elif self.FindRandom(i1):
                return 1
        return 0
 
    def FindMaxNonbound(self,i1,E1):
        '''To find second lagrange multipliers from non-bound.
 
            condition is maximum |E1-E2| of non-bound lagrange multipliers.'''
        i2 = -1
        maxe1e2 = None
        for i in range(0,len(self.alpha)):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                E2 = self.F(i) - self.trainy[i]
                tmp = math.fabs(E1-E2)
                if maxe1e2 == None or maxe1e2 < tmp:
                    maxe1e2 = tmp
                    i2 = i
        if i2 >= 0 and self.StepOnebyOne(i1,i2) :
            return  1              
        return 0
 
    def FindRandomNonbound(self,i1):
        '''To find second lagrange multipliers from non-bound.
 
            condition is random of non-bound lagrange multipliers.'''
        k = random.randint(0,len(self.alpha)-1)
        for i in range(0,len(self.alpha)):
            i2 = (i + k)%len(self.alpha)
            if self.alpha[i2] > 0 and self.alpha[i2] < self.C and self.StepOnebyOne(i1,i2):
                return 1
        return 0
 
    def FindRandom(self,i1):
        '''To find second lagrange multipliers from all.
 
            condition is random one of all lagrange multipliers.'''
        k = random.randint(0,len(self.alpha)-1)
        for i in range(0,len(self.alpha)):
            i2 = (i + k)%len(self.alpha)
            if self.StepOnebyOne(i1,i2):
                return 1
        return 0
 
    def W(self,alpha1new,alpha2newclipped,i1,i2,E1,E2, k11, k22, k12):
        '''To calculate W value.'''
 
        K = self.Kernel()
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.trainy[i1]
        y2 = self.trainy[i2]
        s = y1 * y2
        
        w1 = alpha1new * (y1 * (self.b - E1) + alpha1 * k11 + s * alpha2 * k12)
        w1 += alpha2newclipped * (y2 * (self.b - E2) + alpha2 * k22 + s * alpha1 * k12)
        w1 = w1 - k11 * alpha1new * alpha1new/2 - k22 * alpha2newclipped * alpha2newclipped/2 - s * k12 * alpha1new * alpha2newclipped
        return w1
 
    def StepOnebyOne(self,i1,i2):
        '''To solve two lagrange multipliers problem.
            the algorithm can reference the blog.'''
 
        if i1==i2:
            return 0
 
        #to get kernel function.
        K = self.Kernel()
        
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        alpha1new = -1.0
        alpha2new = -1.0
        alpha2newclipped = -1.0
        y1 = self.trainy[i1]
        y2 = self.trainy[i2]
        s = y1 * y2
        
        key11 = '%s%s%s'%(str(i1), '-', str(i1))
        key22 = '%s%s%s'%(str(i2), '-', str(i2))
        key12 = '%s%s%s'%(str(i1), '-', str(i2))
        key21 = '%s%s%s'%(str(i2), '-', str(i1))
        if self.kcache.has_key(key11):
            k11 = self.kcache[key11]
        else:
            k11 = K(i1,i1)
            self.kcache[key11] = k11    
            
        if self.kcache.has_key(key22):
            k22 = self.kcache[key22]
        else:
            k22 = K(i2,i2)
            self.kcache[key22] = k22
            
        if self.kcache.has_key(key12):
            k12 = self.kcache[key12]
        elif self.kcache.has_key(key21):
            k12 = self.kcache[key21]
        else:
            k12 = K(i1,i2)
            self.kcache[key12] = k12       
        
        eta = k11 + k22 - 2 * k12
        
        E1 = self.F(i1) - y1        
        E2 = self.F(i2) - y2                
 
        #to calucate bound.
        L = 0.0
        H = 0.0
        if y1*y2 == -1:
            gamma = alpha2 - alpha1
            if gamma > 0:
                L = gamma
                H = self.C
            else:
                L = 0
                H = self.C + gamma            
 
        if y1*y2 == 1:
            gamma = alpha2 + alpha1
            if gamma - self.C > 0:
                L = gamma - self.C
                H = self.C
            else:
                L = 0
                H = gamma
        if H == L:
            return 0
        #------------------------begin to move lagrange multipliers.----------------------------
        if -eta < 0:
            #to calculate apha2's new value
            alpha2new = alpha2 + y2 * (E1 - E2)/eta
            
            if alpha2new < L:
                alpha2newclipped = L
            elif alpha2new > H:
                 alpha2newclipped = H
            else:
                alpha2newclipped = alpha2new
        else:            
            w1 = self.W(alpha1 + s * (alpha2 - L),L,i1,i2,E1,E2, k11, k22, k12)
            w2 = self.W(alpha1 + s * (alpha2 - H),H,i1,i2,E1,E2, k11, k22, k12)
            if w1 - w2 > self.accuracy:
                alpha2newclipped = L
            elif w2 - w1 > self.accuracy:
                alpha2newclipped = H
            else:
                alpha2newclipped = alpha2  
        
        if math.fabs(alpha2newclipped - alpha2) < self.accuracy * (alpha2newclipped + alpha2 + self.accuracy):
            return 0
        
        alpha1new = alpha1 + s * (alpha2 - alpha2newclipped)
        if alpha1new < 0:
            alpha2newclipped += s * alpha1new
            alpha1new = 0
        elif alpha1new > self.C:
            alpha2newclipped += s * (alpha1new - self.C)
            alpha1new = self.C
        #------------------------end   to move lagrange multipliers.----------------------------
        if alpha1new > 0 and alpha1new < self.C:
            self.b += (alpha1-alpha1new) * y1 * k11 + (alpha2 - alpha2newclipped) * y2 *k12 - E1
        elif alpha2newclipped > 0 and alpha2newclipped < self.C:
            self.b += (alpha1-alpha1new) * y1 * k12 + (alpha2 - alpha2newclipped) * y2 *k22 - E2
        else:
            b1 = (alpha1-alpha1new) * y1 * k11 + (alpha2 - alpha2newclipped) * y2 *k12 - E1 + self.b
            b2 = (alpha1-alpha1new) * y1 * k12 + (alpha2 - alpha2newclipped) * y2 *k22 - E2 + self.b
            self.b = (b1 + b2)/2
        
        if self.kernelName == 'Linear':
            for j in range(0,self.trainx.nCol):
                self.w[j] += (alpha1new - alpha1) * y1 * self.trainx.Get(i1,j) + (alpha2newclipped - alpha2) * y2 * self.trainx.Get(i2,j)
                
        self.alpha[i1] = alpha1new
        self.alpha[i2] = alpha2newclipped
        
        print 'a', i1, '=',alpha1new,'a', i2,'=', alpha2newclipped
        return 1        
       
    def Train(self,trainx,trainy):
        '''To train samples.
 
            self.trainx is training matrix and self.trainy is classifying label'''
 
        self.trainx = trainx
        self.trainy = trainy
        
        if len(self.trainy) != self.trainx.nRow:
            print "ERROR!, x.nRow should == len(y)"
            return 0
            
        numChanged = 0;
        examineAll = 1;
        #to initialize all lagrange multipiers with zero.
        for i in range(0,self.trainx.nRow):
            self.alpha.append(0.0)
        #to initialize w with zero.
        for j in range(0,self.trainx.nCol):
            self.w.append(float(0))
 
        while numChanged > 0 or examineAll:
            numChanged=0
            print 'numChanged =', numChanged
            if examineAll:
                for k in range(0,self.trainx.nRow):
                    numChanged += self.examineExample(k);#first time or all of lagrange multipiers are abide by KKT conditions then examin all samples.
            else:
                for k in range(0,self.trainx.nRow):
                    if self.alpha[k] !=0 and self.alpha[k] != self.C:
                        numChanged += self.examineExample(k);#to examin all non-bound lagrange multipliers
          
            if(examineAll == 1):
                examineAll = 0
            elif(numChanged == 0):
                examineAll = 1
        else:
            #store support vector machine.                
            self.alphay = []
            self.index = []
            for i in range(0,len(self.alpha)):
                if self.alpha[i] > 0:
                    self.index.append(i)
                    self.alphay.append(self.alpha[i] * self.trainy[i])
                    
            self.sv = [[0 for j in range(self.trainx.nCol)]  for i in range(len(self.index))]
                
            for i in range(0, len(self.index)):
                for j in range(0,self.trainx.nCol):
                    self.sv[i][j] = self.trainx.Get(self.index[i], j)
                
            #dump model path
            f = open(self.modelPath, "w")
            modelStr = pickle.dumps([self.alphay, self.sv, self.b, self.w], 1)
            pickle.dump(modelStr, f)
            f.close()   
            
            self.istrained = True
            
    def Test(self,testx,testy):
        '''To test samples.
 
            self.testx is training matrix and self.testy is classifying label'''    
 
        #check parameter
        if (not self.istrained):
            print "Error!, not trained!"
            return False
        if (testx.nRow != len(testy)):
            print "Error! testx.nRow should == len(testy)"
            return False
            
        self.trainx = testx
        self.trainy = testy
        correct = 0.0
        for i in range(0, self.trainx.nRow):
            fxi = 0.0
            rowvector = [self.trainx.Get(i, k) for k in range(0, self.trainx.nCol)]
 
            if self.kernelName == 'Linear':
                fxi += self.KernelVector(self.w, rowvector) + self.b
            else:
                for j in range(0, len(self.alphay)):                  
                    fxi += self.alphay[j] * self.KernelVector(self.sv[j], rowvector) 
                fxi += self.b
                     
            if fxi * self.trainy[i] >= 0:
                correct +=1
            
            print 'output is', fxi, 'label is', self.trainy[i]
            
        print 'acu=', correct/len(self.trainy)