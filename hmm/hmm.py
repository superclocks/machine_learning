"""
Copyright ChaoZhong superclocks@163.com
"""
import math
import random
import numpy as np
from logprobs import  * #from my code
from tables import OneDTable,TwoDTable#from my code
from str2idmap import Str2IdMap
from sys import stderr

#A transition between two Hmm nodes.
class Transition():
	def __init__(self, from_node, to_node, obs ):
		self._from = from_node
		self._to = to_node
		self._obs = obs
		if(self._from and self._to):
			self._from.outs().append(self)
			self._to.ins().append(self)
#A node in an Hmm object.
class HmmNode():
	def __init__(self, time, state, hmm):
		self._time = time #The time slot for this node.
		self._state = state #The hmm that this node belongs to
		self._logAlpha = 0 #alpha_t(s) = P(e_1:t, x_t=s);
		self._logBeta = 0 #beta_t(s)  = P(e_t+1:T |x_t=s);	 	
		self._psi = 0 #the last transition of the most probable path that reaches this node
		self._hmm = hmm
		
		self._ins = list() #incoming transitions
		self._outs = list() #out going transitions
		
	def time(self):
		return self._time
	def state(self):
		return self._state
	def setLogAlpha(self, logAlpha):
		self._logAlpha = logAlpha
	def getLogAlpha(self):
		return self._logAlpha
	def setLogBeta(self, logBeta):
		self._logBeta = logBeta
	def getLogBeta(self):
		return self._logBeta
	def setPsi(self, psi):
		self._psi = psi
	def getPsi(self):
		return self._psi
	def ins(self):
		return self._ins
	def outs(self):
		return self._outs
	def _print(self):
		print('HmmNode')
# Pseudo Counts 
class PseudoCounts():
	def __init__(self):
		self._stateCount = OneDTable()
		self._transCount = TwoDTable()
		self._emitCount = TwoDTable()
	def getStateCount(self):
		return self._stateCount
	def getTransCount(self):
		return self._transCount
	def getEmitCount(self):
		return self._emitCount
	def _print(self, str2id):
		print('TRANSTION \n')
		#self._transCount.save()
#The possible states at a particular time slot.
class TimeSlot(list): #save HmmNode
	def __init__(self):
		pass
	
class Hmm:
	def __init__(self,init_state = 0, min_log_prob = 0.0000001 ):
		self._init_state = init_state #the initial state
		self._transition =  TwoDTable()#transition probablity
		self._emission = TwoDTable() #emission probablity
		self._str2id = Str2IdMap()  #mapping between strings and integers
		self._time_slots = list() # the time slots
		self._min_log_prob = min_log_prob 
	def loadProbs(self, path):
		''' 
		Read the transition and emission probability tables from the
		files NAME.trans and NAME.emit, where NAME is the value of the
		variable name.
		'''  
		trans_file_path = path + '.trans'
		trans_prob_reader = file(trans_file_path, 'r')
		init_state = trans_prob_reader.readline().split('\n')[0]
		self._init_state = self._str2id.getId(init_state)
		self._transition.load(trans_prob_reader, self._str2id)
		emit_file_path = path + '.emit'
		emit_prob_reader = file(emit_file_path, 'r')
		self._emission.load(emit_prob_reader, self._str2id)
	
	def readSeqs(self, input_file, sequences):
		''' Read the training data from the input stream. Each line in the
		      input stream is an observation sequence. 
		'''	
		while(True):
			ele_set = list()
			line = input_file.readline()
			if(line =='' or line == '\n'):
				break
			ele_in_line = line.split('\n')[0].split(' ')
			for ele in ele_in_line:
				if(ele != ' '):
					ele_set.append(self.getId(ele))
			sequences.append(ele_set)
	'''
	Conversion between the integer id and string form of states and
	observations.
	'''
	def getId(self, str):
		return self._str2id.getId(str)
	def getStr(self, id):
		return self._str2id.getStr(id)
	def addObservation(self, o):
		stateIds = list()
		cntx = self._emission.getCntx(o)
		if cntx == 0:
			keys = self._emission.keys()
			for key in keys:
				stateIds.append(key)
		else:
			for ele in cntx:
				stateIds.append(ele)
			
		if (len(self._time_slots) == 0):
			t0 = TimeSlot()
			t0.append(HmmNode(0, self._init_state,self))
			self._time_slots.append(t0)
		ts = TimeSlot()
		time = len(self._time_slots)
		for i in range(0, len(stateIds)):
			node = HmmNode(time, stateIds[i] , self)
			ts.append(node)
			prev = self._time_slots[time - 1]
			for it in prev:
				possibleSrc = self._transition.getCntx(node.state())
				if(len(possibleSrc) > 0 and possibleSrc.__contains__(it.state())):
					Transition(it , node, o)
		self._time_slots.append(ts)	
	def getTransProb(self, trans):
		return self._transition.get(trans._to.state(), trans._from.state())
	def getEmitProb(self, trans):
		return self._emission.get(trans._obs, trans._to.state())
	#compute the forward probabilities P(e_1:t, X_t=s)
	def forward(self):
		#computer forward probabilities at time 0
		t0 = self._time_slots[0] #TimeSlot
		init = t0[0] #HmmNode
		init.setLogAlpha(0.0)
		#computer forward probabilities at time t using the alpha values for time t-1
		for t in range(1, len(self._time_slots)):
			ts = self._time_slots[t] #get TimeSlot object at time t
			for it in ts: #it is list() type saved the HmmNode
				ins = it.ins() #get Transition list
				log_probs = list()
				for trans in ins:
					log_prob = trans._from.getLogAlpha() + self.getTransProb(trans) + self.getEmitProb(trans)
					log_probs.append(log_prob)
				it.setLogAlpha(sumLogProb1(log_probs))
	#compute the backward probabilities P(e_t+1:T | X_t=s)
	def backward(self):
		T = len(self._time_slots) - 1
		if(T < 1):  #no observation
			return
		time = range(0, T + 1)
		time.reverse()
		for t in time:
			ts = self._time_slots[t]
			for it in ts:
				node = it
				if t ==T:
					node.setLogBeta(0.0)
				else:
					outs = node.outs()
					log_probs = list()
					for i in range(0, len(outs)):
						trans = outs[i]
						log_prob = trans._to.getLogBeta() + \
						self.getTransProb(trans) + self.getEmitProb(trans)
						log_probs.append(log_prob)
					node.setLogBeta(sumLogProb1(log_probs))
	'''
	Accumulate pseudo counts using the BaumWelch algorithm.  The
	return value is the probability of the observations according to
	the current model. 
	'''
	def getPseudoCounts(self, counts):
		p_of_obs = self.obsProb()
		self.backward()
		# Compute the pseudo counts of transitions, emissions, and initializations
		for t in range(0, len(self._time_slots)):
			ts = self._time_slots[t]  #get TimeSlot object at time t
			'''
			P(X_t=s|e_1:T) = alpha_s(t)*beta_s(t)/P(e_t+1:T|e_1:t)
			The value sum below is log P(e_t+1:T|e_1:t)		
			'''			
			log_probs = list()
			for it in ts: #it equ TimeSlot
				log_probs.append(it.getLogAlpha() + it.getLogBeta())
			_sum = sumLogProb1(log_probs)
			#add the pseudo counts into counts
			for it in ts:
				node = it #HmmNode
				#stateCount=P(X_t=s|e_1:T)
				state_count = node.getLogAlpha() + node.getLogBeta() - _sum
				counts.getStateCount().add(node.state(), state_count)
				ins = node.ins() # vector<Transition*>
				for k in range(0, len(ins)):
					trans = ins[k] #Transition*
					_from = trans._from #HmmNode
					trans_count = _from.getLogAlpha() + self.getTransProb(trans) + self.getEmitProb(trans) + node.getLogBeta() - p_of_obs
					counts.getEmitCount().add(node.state(),trans._obs,trans_count)
				outs = node.outs() #vector<Transition*>
				for k in range(0, len(outs)):
					trans = outs[k] #Transition
					to = trans._to #HmmNode
					trans_count = node.getLogAlpha() + self.getTransProb(trans) + self.getEmitProb(trans)+to.getLogBeta() - p_of_obs; 
					counts.getTransCount().add(node.state(), to.state(), trans_count)
		return p_of_obs
	'''
	Find the state sequence (a path) that has the maximum
	probability given the sequence of observations:
	max_{x_1:T} P(x_1:T | e_1:T);
	The return value is the logarithm of the joint probability 
	of the state sequence and the observation sequence:
	log P(x_1:T, e_1:T)
	'''
	def viterbi(self, path):
		#set nodes at time 0 according to initial probabilities.
		ts = self._time_slots[0] #TimeSlot
		init = ts[0] #HmmNode
		init.setLogAlpha(0.0)
		#find the best path up to path t.
		for t in range(1, len(self._time_slots)):
			ts = self._time_slots[t] #ts is vector saved HmmNode
			for it in ts: #it is a HmmNode
				node = it
				ins = node.ins() #ins is vector saved Transition
				max_prob = -1e20
				best_trans = 0 #best_trans is Transition object
				for i in range(0, len(ins)):
					trans = ins[i]
					log_prob = trans._from.getLogAlpha() + \
					         self.getTransProb(trans) + self.getEmitProb(trans)
					if(best_trans == 0 or max_prob < log_prob):
						best_trans = trans
						max_prob = log_prob
				node.setLogAlpha(max_prob) #store the highest probability in logAlpha
				node.setPsi(best_trans) #store the best transition in psi
		#Find the best node at time T. It will be the last node in the best path
		ts = self._time_slots[len(self._time_slots) - 1]
		best = 0 #HmmNode*
		for it in ts:
			node = it
			if (best == 0 or best.getLogAlpha() < node.getLogAlpha()):
				best = node
		#retrieve the nodes in the best path	
		nd = best
		while(nd):
			if(nd.getPsi()):
				path.append(nd.getPsi())
				nd = nd.getPsi()._from
			else:
				nd = 0
		#reverse the path
		i = 0
		j = len(path) - 1
		while(True):
			tmp = path[i]
			path[i] = path[j]
			path[j] = tmp
			if(i >= j):
				break
			i = i + 1
			j = j - 1
		return best.getLogAlpha()
		
	def obsProb(self):
		#return the logarithm of the observation sequence: log P(e_1:T) 
		if(len(self._time_slots) < 1):
			return 1
		self.forward()
		last = self._time_slots[len(self._time_slots) - 1]
		alphaT = list()
		for it in last:
			alphaT.append(it.getLogAlpha())
		return sumLogProb1(alphaT)
	#Clear all time slots to get ready to deal with another sequence. 
	def reset(self):
		for t in range(0, len(self._time_slots)):
			self._time_slots.pop()
	def updateProbs(self, counts):
		self._transition.clear()
		self._emission.clear()
		keys = counts.getTransCount().keys()
		for i in keys:
			_from = i
			from_count = counts.getStateCount().getValue(_from)
			cnts = counts.getTransCount()[i]
			cnts_keys = cnts.keys()
			for j in cnts_keys:
				self._transition.add(_from, j, cnts[j] - from_count)	
		keys = counts.getEmitCount().keys()
		for s in keys:
			state = s
			state_count = counts.getStateCount().get(state)
			cnts = counts.getEmitCount()[s]
			cnts_keys = cnts.keys()
			for o in cnts_keys:
				self._emission.add(state, o, cnts[o] - state_count)
	'''
	Train the model with the given observation sequences using the
	Baum-Welch algorithm. 
	'''
	def baumWelch(self, sequence, max_iterations):
		'''Train the model with the given observation sequences using the
		    Baum-Welch algorithm.
		   '''
		print  'Training with Baum-Welch for up to %d  iterations, using %d sequences.' %(max_iterations, len(sequence))
		prev_total_log_prob = 0
		for k in range(0, max_iterations):
			counts = PseudoCounts()
			total_log_prob = 0
			for i in range(0, len(sequence)):
				seq = sequence[i]
				for j in range(0, len(seq)):
					self.addObservation(seq[j])
				#accumulate the pseudo counts
				total_log_prob += self.getPseudoCounts(counts)
				self.reset()
				if((i+1) % 1000 == 0):
					print('Processed %d sequences' %(i+1) )
			print('Iteration %d total_log_prob = %f' %(k, total_log_prob ))
			if((prev_total_log_prob != 0) and (total_log_prob - prev_total_log_prob < 1)):
				break
			else:
				prev_total_log_prob = total_log_prob
			self.updateProbs(counts)
	def saveProbs(self, name):
		if(name == ''):
			stderr.write('transition probalities: \n')
			self._transition.save(stderr, self._str2id)
			stderr.write('-----------------------\n')
			stderr.write('emission probabilities: \n')
			self._emission.save(stderr, self._str2id)
		else:
			s = name + '.trans'
			trans_prob_writer = file(s, 'w')
			trans_prob_writer.write(self._str2id.getStr(self._init_state))
			trans_prob_writer.write('\n')
			self._transition.save(trans_prob_writer,self._str2id)
			
			s = name + '.emit'
			emit_prob_writer = file(s, 'w')
			self._emission.save(emit_prob_writer, self._str2id)
def testTraining():
	hmm = Hmm()
	hmm.loadProbs('./phone/phone-init1')
	input_reader = file('./phone/phone.train', 'r')
	sequences = list()
	hmm.readSeqs(input_reader,sequences)
	hmm.baumWelch(sequences, 10)
	hmm.saveProbs('./phone/rphone-init1')
def testPrediction():
	hmm = Hmm()
	hmm.loadProbs('./phone/pos/pos')
	input_reader = file('./phone/pos/phone.train', 'r')
	#hmm.loadProbs('./phone/phone-init1')
	#input_reader = file('./phone/phone.train', 'r')
	sequences = list()
	hmm.readSeqs(input_reader,sequences)
	for i in range(0, len(sequences)):
		seq = sequences[i]
		for j in range(0, len(seq)):
			hmm.addObservation(seq[j])
		path = list()
		joint_prob = hmm.viterbi(path)
		print('P(path)=%f ' %exp(joint_prob - hmm.obsProb() ))
		print('path:  ')
		for j in range(0, len(path)):
			trans = path[j]
			if(trans == 0):
				continue
			print('%s \t %s' %(hmm.getStr(trans._obs) , hmm.getStr(trans._to.state())))
		hmm.reset()
if __name__ == "__main__":
	testPrediction()
	testTraining()
	