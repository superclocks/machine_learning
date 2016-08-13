import numpy as np
from math import exp, log

def sumLogProb2(log_prob1, log_prob2):
    if(np.isinf(log_prob1) and np.isinf(log_prob2)):
	return log_prob1
    elif(log_prob1 > log_prob2):
	return log_prob1 + log(1 + exp(log_prob2 - log_prob1))
    else:
	return log_prob2 + log(1 + exp(log_prob1 - log_prob2))
    
def sumLogProb1(log_probs):
    _max = 0
    for i in range(0, len(log_probs)):
	if(i == 0 or log_probs[i] > _max):
	    _max = log_probs[i]
	    
    if(np.isinf(_max)):
	return _max
    p = 0.0
    for i in range(0, len(log_probs)):
	p += exp(log_probs[i] - _max)
    if (p == 0):
	return -1e20
    else:
	return _max + log(p)


