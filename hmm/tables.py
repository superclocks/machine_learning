import str2idmap
import math
from logprobs import *

SMOOTHEDZEROCOUNT=-40
class OneDTable(dict):
    def __init__(self):
	self._smoothed_zero_count = SMOOTHEDZEROCOUNT
    def smoothedZeroCount(self):
	return self._smoothed_zero_count
    
    def getValue(self, event):
	if(self.has_key(event) == False):
	    return self.smoothedZeroCount()
	else:
	    return self.get(event)
    def add(self, event, count):
	if(self.has_key(event) == False):
	    self[event] = count #
	else:
	    self[event] = sumLogProb2(self.get(event), count)
    def rand(self, next):
	p = random.uniform(0, 0x7fff) / 0x7fff
	total = 0
	for key in self.keys:
	    total = total + exp(self.get(key))
	    if(total >= p):
		next = key
		return [next, True]
	return [next, False]
	
class TwoDTable(dict):
    def __init__(self):
	self._possibleContexts = {}
	self._backoff = OneDTable()
    def get(self, event, context):
	if(self.has_key(context) == False):
	    return self._backoff.getValue(event)
	else:
	    return self[context].getValue(event)
    def add(self, context, event, count):
	entry = OneDTable()
	if(self.has_key(context) == False):
	    self[context] = entry
	else:
	    entry = self[context]
	entry.add(event, count)
	#
	possCntx = set()
	if(self._possibleContexts.has_key(event) == False):    
	    self._possibleContexts[event] = possCntx
	else:
	    possCntx = self._possibleContexts[event]
	possCntx.add(context)
    def getCntx(self, event):
	if(self._possibleContexts.has_key(event) == False):
	    return 0
	else:
	    return self._possibleContexts[event]
    def load(self, input_file, str2id):
	while(True):
	    line = input_file.readline()
	    if(line == '\n' or line == ''):
		break	    
	    eles = line.split(' ')
	    if(len(eles) < 3):
		eles = line.split('\t')
	    p = []
	    for ele in eles:
		if(ele != '\n'):
		    p.append(ele)
	    if(float(p[2]) > 0.0):
		self.add(str2id.getId(p[0]), str2id.getId(p[1]), math.log(float(p[2])))			
    def save(self, output_file, str2id):
	twodtable_keys = self.keys()
	for t_key in twodtable_keys:
	    vals = self[t_key]
	    onedtable_keys = vals.keys()
	    for o_key in onedtable_keys:
		output_file.write(str2id.getStr(t_key))
		output_file.write(' ')
		output_file.write(str2id.getStr(o_key))
		output_file.write(' ')
		output_file.write(str(exp(vals[o_key])))
		output_file.write('\n')	
    def rand(self, curr, next):
	if(self.has_key(curr) == False):
	    return False
	val = self.get(curr)
	return val.rand(next)